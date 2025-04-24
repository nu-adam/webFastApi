from flask import Flask, request, jsonify
import os
from datetime import datetime
from moviepy import VideoFileClip
from model.app import main as analyze_clip
from flask import Response
import json
from flask_cors import CORS
from threading import Lock
import time

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})
# Get absolute paths for the upload and split directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
SPLIT_FOLDER = os.path.join(BASE_DIR, 'splits')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPLIT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPLIT_FOLDER'] = SPLIT_FOLDER

# Add these at the top of your file
progress_data = {
    'message': 'Waiting for upload to start...',
    'progress': 0,
    'error': None,
    'complete': False
}
progress_lock = Lock()

@app.route('/upload-progress', methods=['GET'])
def upload_progress():
    def generate():
        last_sent = 0
        while True:
            with progress_lock:
                current_data = progress_data.copy()
            
            # Only send if there's new data or it's complete
            if current_data['progress'] != last_sent or current_data['complete']:
                yield f"data: {json.dumps(current_data)}\n\n"
                last_sent = current_data['progress']
                
                if current_data['complete'] or current_data['error']:
                    break
            
            time.sleep(0.5)  # Reduce CPU usage

    return Response(generate(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Reset progress data
        with progress_lock:
            progress_data.update({
                'message': 'Starting upload...',
                'progress': 0,
                'error': None,
                'complete': False
            })

        # Sanitize filename
        original_filename = os.path.basename(video_file.filename)
        if not original_filename:
            with progress_lock:
                progress_data.update({
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
                    progress_data.update({
                        'progress': progress,
                        'message': 'Uploading video...'
                    })

        # Get filename without extension for folder naming
        base_name = os.path.splitext(original_filename)[0]
        clip_output_dir = os.path.join(app.config['SPLIT_FOLDER'], base_name)
        os.makedirs(clip_output_dir, exist_ok=True)

        # Split the video into fixed-length clips
        CLIP_LENGTH = 4  # seconds

        clip = VideoFileClip(save_path)
        duration = int(clip.duration)
        total_clips = (duration + CLIP_LENGTH - 1) // CLIP_LENGTH  # Round up
        part_index = 1
        
        for start_time in range(0, duration, CLIP_LENGTH):
            end_time = min(start_time + CLIP_LENGTH, clip.duration)
            subclip = clip.subclipped(start_time, end_time)
            subclip_path = os.path.join(clip_output_dir, f"{base_name}_part{part_index}.mp4")
            subclip.write_videofile(
                subclip_path, 
                codec='libx264', 
                audio_codec='aac', 
                logger=None,
                threads=4
            )
            part_index += 1
            
            # Update progress (second 50% for splitting)
            progress = 50 + (part_index / total_clips) * 50
            with progress_lock:
                progress_data.update({
                    'progress': progress,
                    'message': f'Splitting video ({part_index}/{total_clips})...',
                    'current_clip': part_index,
                    'total_clips': total_clips
                })
        
        clip.close()
        
        # Send completion data
        with progress_lock:
            progress_data.update({
                'progress': 100,
                'message': 'Upload and split complete!',
                'uploaded_path': save_path,
                'split_folder': clip_output_dir,
                'clips_created': part_index - 1,
                'complete': True
            })

        return jsonify({
            "message": "File uploaded and split successfully",
            "uploaded_path": save_path,
            "split_folder": clip_output_dir,
            "clips_created": part_index - 1
        })
            
    except Exception as e:
        with progress_lock:
            progress_data.update({
                'error': str(e),
                'complete': True
            })
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-clips', methods=['GET'])
def analyze_clips():
    # Get split_folder from query parameters instead of JSON body
    split_folder = request.args.get('split_folder')
    
    if not split_folder:
        return jsonify({"error": "No split_folder provided"}), 400
    
    if not os.path.isdir(split_folder):
        return jsonify({"error": "Provided split_folder does not exist"}), 400
    
    video_files = sorted([
        os.path.join(split_folder, f) 
        for f in os.listdir(split_folder) 
        if f.endswith(('.mp4', '.mov', '.avi', '.mkv'))
    ])
    
    total_clips = len(video_files)
    if total_clips == 0:
        return jsonify({"error": "No video clips found in the directory"}), 400

    def generate():
        try:
            for index, clip_path in enumerate(video_files, 1):
                try:
                    analysis_result = analyze_clip(clip_path)
                    
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
                    
            # Send completion event
            yield "event: complete\ndata: {}\n\n"
            
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

if __name__ == '__main__':
    app.run(debug=True)