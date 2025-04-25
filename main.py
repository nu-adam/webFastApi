from flask import Flask, request, jsonify
import os
import cv2  # OpenCV instead of MoviePy
import numpy as np
from datetime import datetime, timedelta
from model.app import main as analyze_clip
from flask import Response
import json
from flask_cors import CORS
from model.models import db, User, Video
from model.auth import init_auth, token_required
import secrets
from threading import Lock
import time

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['GOOGLE_CLIENT_ID'] = os.environ.get('GOOGLE_CLIENT_ID', '')  # Set this in your environment

# Setup CORS
CORS(app, resources={
    r"/*": {"origins": "*"}, 
    r"/analyze-clips": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize database
db.init_app(app)

# Initialize authentication
init_auth(app)

# Get absolute paths for the upload and split directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
SPLIT_FOLDER = os.path.join(BASE_DIR, 'splits')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPLIT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPLIT_FOLDER'] = SPLIT_FOLDER

# Create database tables
with app.app_context():
    db.create_all()

def split_video_with_opencv(input_path, output_dir, base_name, clip_length=4):
    """
    Split video into fixed-length clips using OpenCV
    """
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate frames per clip
    frames_per_clip = int(clip_length * fps)
    
    # Initialize variables
    part_index = 1
    frame_count = 0
    current_writer = None
    
    # Loop through the video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start a new clip if needed
        if frame_count % frames_per_clip == 0:
            # Close previous writer if exists
            if current_writer is not None:
                current_writer.release()
            
            # Create a new output file
            output_path = os.path.join(output_dir, f"{base_name}_part{part_index}.mp4")
            
            # Define the codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
            current_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            part_index += 1
        
        # Write the frame to current clip
        if current_writer is not None:
            current_writer.write(frame)
        
        frame_count += 1
    
    # Release resources
    if current_writer is not None:
        current_writer.release()
    cap.release()
    
    return part_index - 1  # Return number of clips created

# Global progress tracker
upload_progress = {
    'progress': 0,
    'message': '',
    'error': None,
    'complete': False,
    'result': None
}
progress_lock = Lock()

@app.route('/upload-progress', methods=['GET'])
def upload_progress_endpoint():
    def generate():
        last_sent = None
        while True:
            with progress_lock:
                current_data = upload_progress.copy()
            
            # Only send if there's new data
            if current_data != last_sent:
                yield f"data: {json.dumps(current_data)}\n\n"
                last_sent = current_data
                
                if current_data['complete'] or current_data['error']:
                    break
            
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
@token_required
def upload_file(current_user):
    # Reset progress data
    with progress_lock:
        upload_progress.update({
            'progress': 0,
            'message': 'Starting upload...',
            'error': None,
            'complete': False,
            'result': None
        })

    if 'video' not in request.files:
        with progress_lock:
            upload_progress.update({
                'error': 'No video file in request',
                'complete': True
            })
        return jsonify({"error": "No video file in request"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        with progress_lock:
            upload_progress.update({
                'error': 'No selected file',
                'complete': True
            })
        return jsonify({"error": "No selected file"}), 400

    try:
        # Sanitize filename
        original_filename = os.path.basename(video_file.filename)
        if not original_filename:
            with progress_lock:
                upload_progress.update({
                    'error': 'Invalid filename',
                    'complete': True
                })
            return jsonify({"error": "Invalid filename"}), 400

        # Add timestamp prefix to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{original_filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file in chunks to track progress
        total_size = int(request.headers.get('Content-Length', 0))
        chunk_size = 4096  # 4KB chunks
        bytes_read = 0
        
        with open(save_path, 'wb') as f:
            while True:
                chunk = video_file.stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_read += len(chunk)
                progress = min(50, (bytes_read / total_size) * 50)  # First 50% for upload
                with progress_lock:
                    upload_progress.update({
                        'progress': progress,
                        'message': 'Uploading video...'
                    })

        # Get filename without extension for folder naming
        base_name = os.path.splitext(original_filename)[0]
        clip_output_dir = os.path.join(app.config['SPLIT_FOLDER'], base_name)
        os.makedirs(clip_output_dir, exist_ok=True)

        # Split the video into fixed-length clips using OpenCV
        clips_created = split_video_with_opencv(
            save_path, 
            clip_output_dir, 
            base_name, 
            clip_length=4
        )
        
        # Update progress during processing
        with progress_lock:
            upload_progress.update({
                'progress': 75,
                'message': 'Processing video...'
            })

        # Create the video record in database
        new_video = Video(
            user_id=current_user.user_id,
            title=original_filename,
            description=f"Uploaded on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            file_path=save_path
        )
        db.session.add(new_video)
        db.session.commit()
        
        # Final update with results
        with progress_lock:
            upload_progress.update({
                'progress': 100,
                'message': 'Upload and processing complete!',
                'complete': True,
                'result': {
                    "uploaded_path": save_path,
                    "split_folder": clip_output_dir,
                    "clips_created": clips_created,
                    "video_id": new_video.video_id
                }
            })

        return jsonify(upload_progress['result'])
        
    except Exception as e:
        with progress_lock:
            upload_progress.update({
                'error': str(e),
                'complete': True
            })
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-clips', methods=['GET'])
@token_required
def analyze_clips(current_user):
    split_folder = request.args.get('split_folder')
    video_id = request.args.get('video_id')

    if not split_folder:
        return jsonify({"error": "No split_folder provided"}), 400
    
    if not os.path.isdir(split_folder):
        return jsonify({"error": "Provided split_folder does not exist"}), 400
    
    if video_id:
        video = Video.query.get(int(video_id))
        if not video or video.user_id != current_user.user_id:
            return jsonify({"error": "Unauthorized access to video"}), 403

    try:
        video_files = sorted([
            os.path.join(split_folder, f) 
            for f in os.listdir(split_folder) 
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
        ], key=lambda x: int(x.split('_part')[-1].split('.')[0]) if '_part' in x else 0)
        print(f"Video files sorted: {video_files}")
        
        if not video_files:
            return jsonify({"error": "No video clips found"}), 400
    except Exception as e:
        print(f"Error reading clips: {str(e)}")
        return jsonify({"error": f"Error reading clips: {str(e)}"}), 500

    def generate():
        emotion_totals = {e: 0.0 for e in ['happiness', 'anger', 'neutral', 'sadness']}  # Update to match EMOTION_LABELS
        processed_clips = 0
        errors = []
        transcriptions = []  # Store transcriptions

        try:
            for idx, clip_path in enumerate(video_files, 1):
                try:
                    if not os.path.exists(clip_path):
                        raise Exception(f"Clip not found: {clip_path}")
                    
                    cap = cv2.VideoCapture(clip_path)
                    if not cap.isOpened():
                        raise Exception(f"Cannot open clip: {clip_path}")
                    cap.release()
                    
                    result = json.loads(analyze_clip(clip_path))  # Parse JSON string to dict
                    print(f"analyze_clip result for {clip_path}: {result}")
                    if 'error' in result:
                        raise Exception(result['error'])
                    
                    probabilities = result.get('probabilities', {e: 0.0 for e in emotion_totals})
                    max_emotion = max(probabilities, key=probabilities.get)
                    formatted_result = {
                        'emotion': result.get('emotion', max_emotion),
                        'confidence': result.get('confidence', probabilities[max_emotion]),
                        'probabilities': probabilities,
                        'transcribed_text': result.get('transcribed_text', '')  # Include transcription
                    }
                    
                    for emotion in emotion_totals:
                        if emotion in probabilities:
                            emotion_totals[emotion] += probabilities[emotion]
                    processed_clips += 1
                    transcriptions.append(formatted_result['transcribed_text'])

                    yield f"data: {json.dumps({
                        'current': idx,
                        'total': len(video_files),
                        'clip_path': clip_path,
                        'result': formatted_result
                    })}\n\n"

                except Exception as e:
                    errors.append(f"Clip {idx}: {str(e)}")
                    print(f"Error processing clip {idx}: {str(e)}")
                    yield f"data: {json.dumps({
                        'current': idx,
                        'total': len(video_files),
                        'clip_path': clip_path,
                        'error': str(e)
                    })}\n\n"
                    continue

            final_scores = {e: t/processed_clips if processed_clips > 0 else 0.0 for e, t in emotion_totals.items()}
            
            if video_id:
                try:
                    video = Video.query.get(int(video_id))
                    if video:
                        for emotion, score in final_scores.items():
                            setattr(video, emotion, score)
                        video.transcription = " ".join(transcriptions)  # Store combined transcription
                        db.session.commit()
                except Exception as e:
                    print(f"Database update error: {str(e)}")

            yield f"event: complete\ndata: {json.dumps({
                'emotion_scores': final_scores,
                'processed_clips': processed_clips,
                'errors': errors,
                'transcription': " ".join(transcriptions)  # Include in final response
            })}\n\n"

        except Exception as e:
            errors.append(f"Analysis failed: {str(e)}")
            print(f"Analysis failed: {str(e)}")
            yield f"data: {json.dumps({
                'error': f"Analysis failed: {str(e)}",
                'current': processed_clips,
                'total': len(video_files),
                'errors': errors
            })}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/videos', methods=['GET'])
@token_required
def get_user_videos(current_user):
    videos = Video.query.filter_by(user_id=current_user.user_id).order_by(Video.created_at.desc()).all()
    
    video_list = []
    for video in videos:
        video_list.append({
            'id': video.video_id,
            'title': video.title,
            'description': video.description,
            'created_at': video.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'emotions': {
                'happiness': video.happiness,
                'frustration': video.frustration,
                'anger': video.anger,
                'sadness': video.sadness,
                'neutral': video.neutral,
                'excited': video.excited
            }
        })
    
    return jsonify({"videos": video_list})

if __name__ == '__main__':
    app.run(debug=True)