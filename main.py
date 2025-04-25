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
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from model.auth import GOOGLE_CLIENT_ID

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['GOOGLE_CLIENT_ID'] = os.environ.get('GOOGLE_CLIENT_ID', '')  # Set this in your environment

# Setup CORS
CORS(app, resources={r"/*": {"origins": "*"}})

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

@app.route('/upload', methods=['POST'])
@token_required
def upload_file(current_user):
    if 'video' not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Sanitize filename
    original_filename = os.path.basename(video_file.filename)
    if not original_filename:
        return jsonify({"error": "Invalid filename"}), 400

    # Add timestamp prefix to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{original_filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        video_file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    # Get filename without extension for folder naming
    base_name = os.path.splitext(original_filename)[0]
    clip_output_dir = os.path.join(app.config['SPLIT_FOLDER'], base_name)
    os.makedirs(clip_output_dir, exist_ok=True)

    # Split the video into fixed-length clips using OpenCV
    try:
        clips_created = split_video_with_opencv(
            save_path, 
            clip_output_dir, 
            base_name, 
            clip_length=4
        )
        
        # Create the video record in database
        new_video = Video(
            user_id=current_user.user_id,
            title=original_filename,
            description=f"Uploaded on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            file_path=save_path
        )
        db.session.add(new_video)
        db.session.commit()
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "message": "File uploaded and split successfully",
        "uploaded_path": save_path,
        "split_folder": clip_output_dir,
        "clips_created": clips_created,
        "video_id": new_video.video_id
    })

@app.route('/analyze-clips', methods=['GET'])
@token_required
def analyze_clips(current_user=None):
    # Get split_folder from query parameters instead of JSON body
    split_folder = request.args.get('split_folder')
    video_id = request.args.get('video_id')
    token = request.args.get('token')
    
    if not current_user and token:
        try:
            # Verify token
            idinfo = id_token.verify_oauth2_token(
                token, 
                google_requests.Request(), 
                GOOGLE_CLIENT_ID
            )
            
            # Get user
            google_id = idinfo['sub']
            current_user = User.query.filter_by(google_id=google_id).first()
            
            if not current_user:
                return jsonify({"error": "Unauthorized: User not found"}), 401
        except Exception as e:
            return jsonify({"error": f"Authentication failed: {str(e)}"}), 401
    if not split_folder:
        return jsonify({"error": "No split_folder provided"}), 400
    
    if not os.path.isdir(split_folder):
        return jsonify({"error": "Provided split_folder does not exist"}), 400
    
    # Verify that the video belongs to the current user
    if video_id:
        video = Video.query.get(int(video_id))
        if not video or video.user_id != current_user.user_id:
            return jsonify({"error": "Unauthorized access to video"}), 403
    
    video_files = sorted([
        os.path.join(split_folder, f) 
        for f in os.listdir(split_folder) 
        if f.endswith(('.mp4', '.mov', '.avi', '.mkv'))
    ])
    
    total_clips = len(video_files)
    if total_clips == 0:
        return jsonify({"error": "No video clips found in the directory"}), 400

    def generate():
        total_emotion_scores = {
            'happiness': 0,
            'frustration': 0,
            'anger': 0,
            'sadness': 0,
            'neutral': 0,
            'excited': 0
        }
        
        try:
            for index, clip_path in enumerate(video_files, 1):
                try:
                    analysis_result = analyze_clip(clip_path)
                    
                    # Accumulate emotion scores
                    for emotion, score in analysis_result.items():
                        if emotion in total_emotion_scores:
                            total_emotion_scores[emotion] += score
                    
                    progress_data = {
                        "current": index,
                        "total": total_clips,
                        "result": analysis_result,
                        "clip_path": clip_path
                    }
                    
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    
                except Exception as e:
                    error_data = {
                        "error": str(e),
                        "clip_path": clip_path,
                        "current": index,
                        "total": total_clips
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            # Calculate average scores
            for emotion in total_emotion_scores:
                if total_clips > 0:
                    total_emotion_scores[emotion] /= total_clips
            
            # Update video record with emotion scores if video_id is provided
            if video_id:
                video = Video.query.get(int(video_id))
                if video and video.user_id == current_user.user_id:
                    for emotion, score in total_emotion_scores.items():
                        setattr(video, emotion, score)
                    db.session.commit()
            
            # Send completion event with final scores
            completion_data = {
                "status": "complete",
                "emotion_scores": total_emotion_scores
            }
            yield f"event: complete\ndata: {json.dumps(completion_data)}\n\n"
            
        except GeneratorExit:
            print("Client disconnected")
        except Exception as e:
            error_data = {
                "error": f"Analysis failed: {str(e)}",
                "current": 0,
                "total": total_clips
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    response = Response(generate(), mimetype='text/event-stream')
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Cache-Control', 'no-cache')
    return response

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