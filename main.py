from flask import Flask, request, jsonify
import os
from datetime import datetime
from moviepy import VideoFileClip
from model.app import main as analyze_clip
from flask import Response
import json
from flas_cors import CORS

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

@app.route('/upload', methods=['POST'])
def upload_file():
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

    # Split the video into fixed-length clips
    CLIP_LENGTH = 4  # seconds

    try:
        clip = VideoFileClip(save_path)
        duration = int(clip.duration)
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
        clip.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "message": "File uploaded and split successfully",
        "uploaded_path": save_path,
        "split_folder": clip_output_dir,
        "clips_created": part_index - 1
    })

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