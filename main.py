from flask import Flask, request, jsonify, Response
import os
import cv2
import numpy as np
from datetime import datetime
from model.app import main as analyze_clip
import json
from flask_cors import CORS
from model.models import db, User, Video
from model.auth import init_auth, token_required
import secrets
from threading import Lock
import time
import whisper
import nltk
import soundfile as sf
import librosa
import subprocess
import logging as logger

# Download required NLTK data
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading punkt_tab resource...")
    nltk.download('punkt_tab')

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['GOOGLE_CLIENT_ID'] = os.environ.get('GOOGLE_CLIENT_ID', '')

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

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Uploads')
VIDEO_SPLITS_FOLDER = os.path.join(BASE_DIR, 'video_splits')
AUDIO_SPLITS_FOLDER = os.path.join(BASE_DIR, 'audio_splits')
TEXT_SPLITS_FOLDER = os.path.join(BASE_DIR, 'text_splits')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_SPLITS_FOLDER, exist_ok=True)
os.makedirs(AUDIO_SPLITS_FOLDER, exist_ok=True)
os.makedirs(TEXT_SPLITS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_SPLITS_FOLDER'] = VIDEO_SPLITS_FOLDER
app.config['AUDIO_SPLITS_FOLDER'] = AUDIO_SPLITS_FOLDER
app.config['TEXT_SPLITS_FOLDER'] = TEXT_SPLITS_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Create database tables
with app.app_context():
    db.create_all()

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Progress tracker
upload_progress = {
    'progress': 0,
    'message': '',
    'error': None,
    'complete': False,
    'result': None
}
progress_lock = Lock()

def extract_audio_from_video(video_path, output_path):
    """Extract audio from video to WAV file using ffmpeg"""
    try:
        logger.info(f"Extracting audio from {video_path} to {output_path}")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-ac', '1', '-ar', '16000',
            '-acodec', 'pcm_s16le',
            '-loglevel', 'error',
            output_path
        ], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise Exception(f"Audio extraction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio extraction error: {str(e)}")
        raise

def transcribe_and_segment(video_path):
    """Transcribe video and segment into sentences with timestamps"""
    try:
        audio_path = os.path.join(
            app.config['AUDIO_SPLITS_FOLDER'],
            f"temp_{os.path.basename(video_path)}.wav"
        )
        
        extract_audio_from_video(video_path, audio_path)
        logger.info(f"Audio extracted to {audio_path}")

        logger.info("Starting transcription...")
        result = whisper_model.transcribe(audio_path, language="en")
        segments = result["segments"]
        logger.info(f"Transcription complete: {len(segments)} segments")

        sentences = []
        current_sentence = ""
        start_time = None
        
        for segment in segments:
            text = segment["text"].strip()
            if not text:
                continue
                
            segment_sentences = nltk.sent_tokenize(text)
            for sentence in segment_sentences:
                if not sentence:
                    continue
                    
                if not current_sentence:
                    start_time = segment["start"]
                current_sentence += " " + sentence
                
                if sentence.endswith(('.', '!', '?')):
                    sentences.append({
                        "text": current_sentence.strip(),
                        "start_time": start_time,
                        "end_time": segment["end"]
                    })
                    current_sentence = ""
                    start_time = None

        if current_sentence.strip():
            sentences.append({
                "text": current_sentence.strip(),
                "start_time": start_time or segments[-1]["start"],
                "end_time": segments[-1]["end"]
            })

        logger.info(f"Segmented into {len(sentences)} sentences")
        return sentences, result["text"]

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

def split_video_by_timestamps(video_path, output_dir, base_name, sentences):
    """Split video into clips based on sentence timestamps"""
    cap = None
    try:
        logger.info(f"Splitting video {video_path} into clips")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        clip_paths = []
        for idx, sentence in enumerate(sentences, 1):
            start_frame = int(sentence["start_time"] * fps)
            end_frame = int(sentence["end_time"] * fps)
            output_path = os.path.join(output_dir, f"{base_name}_part{idx}.mp4")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise Exception(f"Could not create video writer for {output_path}")

            for _ in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            writer.release()
            clip_paths.append(output_path)
            logger.info(f"Created clip {idx}: {output_path}")

        return clip_paths

    except Exception as e:
        logger.error(f"Video splitting failed: {str(e)}")
        raise
    finally:
        if cap:
            cap.release()

def split_audio_by_timestamps(video_path, output_dir, base_name, sentences):
    """Split audio into clips based on sentence timestamps"""
    y, sr = librosa.load(video_path, sr=16000, mono=True)
    audio_paths = []
    
    for idx, sentence in enumerate(sentences, 1):
        start_time = sentence["start_time"]
        end_time = sentence["end_time"]
        output_path = os.path.join(output_dir, f"{base_name}_part{idx}.wav")
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        clip_audio = y[start_sample:end_sample]
        sf.write(output_path, clip_audio, sr)
        audio_paths.append(output_path)
    
    return audio_paths

@app.route('/upload-progress', methods=['GET'])
def upload_progress_endpoint():
    def generate():
        last_sent = None
        while True:
            with progress_lock:
                current_data = upload_progress.copy()
            
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
    with progress_lock:
        upload_progress.update({
            'progress': 0,
            'message': 'Starting upload...',
            'error': None,
            'complete': False,
            'result': None
        })

    try:
        if 'video' not in request.files:
            raise ValueError("No video file in request")

        video_file = request.files['video']
        if video_file.filename == '':
            raise ValueError("No selected file")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{video_file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with progress_lock:
            upload_progress['message'] = 'Saving video file...'

        video_file.save(save_path)
        logger.info(f"Video saved to {save_path}")

        with progress_lock:
            upload_progress.update({
                'progress': 30,
                'message': 'Transcribing audio...'
            })

        sentences, full_text = transcribe_and_segment(save_path)

        base_name = os.path.splitext(video_file.filename)[0]
        video_split_dir = os.path.join(app.config['VIDEO_SPLITS_FOLDER'], base_name)
        audio_split_dir = os.path.join(app.config['AUDIO_SPLITS_FOLDER'], base_name)
        os.makedirs(video_split_dir, exist_ok=True)
        os.makedirs(audio_split_dir, exist_ok=True)

        with progress_lock:
            upload_progress.update({
                'progress': 50,
                'message': 'Splitting video...'
            })

        video_clip_paths = split_video_by_timestamps(save_path, video_split_dir, base_name, sentences)

        with progress_lock:
            upload_progress.update({
                'progress': 70,
                'message': 'Splitting audio...'
            })

        audio_clip_paths = split_audio_by_timestamps(save_path, audio_split_dir, base_name, sentences)

        text_split_path = os.path.join(app.config['TEXT_SPLITS_FOLDER'], f"{base_name}.json")
        text_data = [{
            "sentence": s["text"],
            "start_time": s["start_time"],
            "end_time": s["end_time"],
            "video_clip": video_clip_paths[idx],
            "audio_clip": audio_clip_paths[idx]
        } for idx, s in enumerate(sentences)]

        with open(text_split_path, 'w') as f:
            json.dump(text_data, f, indent=2)

        new_video = Video(
            user_id=current_user.user_id,
            title=video_file.filename,
            description=f"Uploaded on {datetime.now()}",
            file_path=save_path,
        )
        db.session.add(new_video)
        db.session.commit()

        with progress_lock:
            upload_progress.update({
                'progress': 100,
                'message': 'Processing complete!',
                'complete': True,
                'result': {
                    "video_id": new_video.video_id,
                    "clips_created": len(sentences),
                    "video_split_folder": video_split_dir,
                    "audio_split_folder": audio_split_dir,
                    "text_split_path": text_split_path,
                    "results_path": os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}.json")
                }
            })

        return jsonify(upload_progress['result'])

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        with progress_lock:
            upload_progress.update({
                'error': str(e),
                'complete': True
            })
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-clips', methods=['GET'])
@token_required
def analyze_clips(current_user):
    video_split_folder = request.args.get('video_split_folder')
    audio_split_folder = request.args.get('audio_split_folder')
    text_split_path = request.args.get('text_split_path')
    video_id = request.args.get('video_id')

    if not all([video_split_folder, audio_split_folder, text_split_path]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    if not os.path.isdir(video_split_folder) or not os.path.isdir(audio_split_folder) or not os.path.exists(text_split_path):
        return jsonify({"error": "Provided paths do not exist"}), 400
    
    if video_id:
        video = Video.query.get(int(video_id))
        if not video or video.user_id != current_user.user_id:
            return jsonify({"error": "Unauthorized access to video"}), 403

    try:
        with open(text_split_path, 'r') as f:
            text_data = json.load(f)
        
        clip_data = sorted(
            text_data,
            key=lambda x: int(x["video_clip"].split('_part')[-1].split('.')[0])
        )
        
        if not clip_data:
            return jsonify({"error": "No clips found in text data"}), 400
        
        # Initialize JSON result file
        base_name = os.path.basename(video_split_folder)
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}.json")
        analysis_results = {
            "video_id": video_id,
            "clips": [],
            "emotion_scores": {},
            "transcription": "",
            "processed_clips": 0,
            "errors": []
        }
        with open(result_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)

    except Exception as e:
        return jsonify({"error": f"Error reading text data: {str(e)}"}), 500

    def generate():
        emotion_totals = {e: 0.0 for e in ['happiness', 'anger', 'neutral', 'sadness']}
        processed_clips = 0
        errors = []
        transcriptions = []

        try:
            for idx, clip in enumerate(clip_data, 1):
                try:
                    video_path = clip["video_clip"]
                    audio_path = clip["audio_clip"]
                    text = clip["sentence"]
                    start_time = clip.get("start_time")
                    end_time = clip.get("end_time")
                    
                    if not os.path.exists(video_path) or not os.path.exists(audio_path):
                        raise Exception(f"Clip files missing: {video_path} or {audio_path}")
                    
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        raise Exception(f"Cannot open clip: {video_path}")
                    cap.release()
                    
                    result = json.loads(analyze_clip(video_path, audio_path, text))
                    if 'error' in result:
                        raise Exception(result['error'])
                    
                    probabilities = result.get('probabilities', {e: 0.0 for e in emotion_totals})
                    max_emotion = max(probabilities, key=probabilities.get)
                    formatted_result = {
                        'emotion': result.get('emotion', max_emotion),
                        'confidence': result.get('confidence', probabilities[max_emotion]),
                        'probabilities': probabilities,
                        'transcribed_text': text,
                        'start_time': start_time,
                        'end_time': end_time,
                        'video_clip_path': video_path,
                        'audio_clip_path': audio_path
                    }
                    
                    # Update JSON file
                    with open(result_path, 'r+') as f:
                        data = json.load(f)
                        data['clips'].append(formatted_result)
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()
                    
                    for emotion in emotion_totals:
                        if emotion in probabilities:
                            emotion_totals[emotion] += probabilities[emotion]
                    processed_clips += 1
                    transcriptions.append(text)

                    yield f"data: {json.dumps({
                        'current': idx,
                        'total': len(clip_data),
                        'video_clip_path': video_path,
                        'audio_clip_path': audio_path,
                        'text': text,
                        'start_time': start_time,
                        'end_time': end_time,
                        'result': formatted_result
                    })}\n\n"

                except Exception as e:
                    errors.append(f"Clip {idx}: {str(e)}")
                    with open(result_path, 'r+') as f:
                        data = json.load(f)
                        data['errors'].append(f"Clip {idx}: {str(e)}")
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()
                    
                    yield f"data: {json.dumps({
                        'current': idx,
                        'total': len(clip_data),
                        'video_clip_path': video_path,
                        'audio_clip_path': audio_path,
                        'text': text,
                        'start_time': clip.get('start_time'),
                        'end_time': clip.get('end_time'),
                        'error': str(e)
                    })}\n\n"
                    continue

            final_scores = {e: t/processed_clips if processed_clips > 0 else 0.0 for e, t in emotion_totals.items()}
            
            # Update final results in JSON
            with open(result_path, 'r+') as f:
                data = json.load(f)
                data['emotion_scores'] = final_scores
                data['processed_clips'] = processed_clips
                data['transcription'] = " ".join(transcriptions)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            
            if video_id:
                try:
                    video = Video.query.get(int(video_id))
                    if video:
                        for emotion, score in final_scores.items():
                            setattr(video, emotion, score)
                        video.transcription = " ".join(transcriptions)
                        video.results_path = result_path
                        db.session.commit()
                except Exception as e:
                    print(f"Database update error: {str(e)}")

            yield f"event: complete\ndata: {json.dumps({
                'emotion_scores': final_scores,
                'processed_clips': processed_clips,
                'errors': errors,
                'transcription': " ".join(transcriptions),
                'results_path': result_path
            })}\n\n"

        except Exception as e:
            errors.append(f"Analysis failed: {str(e)}")
            with open(result_path, 'r+') as f:
                data = json.load(f)
                data['errors'].append(f"Analysis failed: {str(e)}")
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            
            yield f"data: {json.dumps({
                'error': f"Analysis failed: {str(e)}",
                'current': processed_clips,
                'total': len(clip_data),
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

@app.route('/user/videos', methods=['GET'])
@token_required
def get_user_videos(current_user):
    videos = Video.query.filter_by(user_id=current_user.user_id).order_by(Video.created_at.desc()).all()
    
    video_list = []
    for video in videos:
        analysis_data = None
        base_name = os.path.splitext(video.title)[0]
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}.json")
        
        # Check if results file exists
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    detailed_results = json.load(f)
                
                analysis_data = {
                    'analysis_date': video.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'average_emotions': {
                        'happiness': detailed_results.get('emotion_scores', {}).get('happiness', video.happiness or 0),
                        'anger': detailed_results.get('emotion_scores', {}).get('anger', video.anger or 0),
                        'neutral': detailed_results.get('emotion_scores', {}).get('neutral', video.neutral or 0),
                        'sadness': detailed_results.get('emotion_scores', {}).get('sadness', video.sadness or 0)
                    },
                    'detailed_results': detailed_results
                }
            except Exception as e:
                logger.error(f"Error reading results file {results_path}: {str(e)}")
        elif any([video.happiness, video.anger, video.neutral, video.sadness]):
            analysis_data = {
                'analysis_date': video.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'average_emotions': {
                    'happiness': video.happiness or 0,
                    'anger': video.anger or 0,
                    'neutral': video.neutral or 0,
                    'sadness': video.sadness or 0
                }
            }
        
        video_list.append({
            'id': video.video_id,
            'title': video.title,
            'description': video.description,
            'created_at': video.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_data': analysis_data
        })
    
    return jsonify({"videos": video_list})

if __name__ == '__main__':
    app.run(debug=True)