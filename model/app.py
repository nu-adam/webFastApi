import sys
import os
import torch
import numpy as np
import cv2
import librosa
from insightface.app import FaceAnalysis
from transformers import RobertaTokenizer, RobertaModel, logging as transformers_logging
import torchvision.transforms as T
from torchvision import models
from torch import nn
from einops import repeat
from datetime import datetime
import json
import warnings
import logging
from contextlib import contextmanager

def get_base_path():
    if getattr(sys, 'frozen', False):
        # For frozen/executable (PyInstaller)
        base_path = os.path.dirname(sys.executable)
    else:
        # For normal Python execution
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    return base_path

BASE_PATH = get_base_path()
MODEL_CHECKPOINT = os.path.join(BASE_PATH, "checkpoints", "best_model_epoch13_6807.pth")

# Configuration
NUM_CLASSES = 4  # Update based on your model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Emotion labels (update based on your training)
EMOTION_LABELS = [
    "anger", "happiness", "neutral", "sadness",
]

# # Suppressing Logging:
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# transformers_logging.set_verbosity_error()
# warnings.filterwarnings("ignore")
# logging.getLogger().setLevel(logging.ERROR)
# @contextmanager
# def suppress_all():
#     with open(os.devnull, 'w') as devnull:
#         old_stdout = sys.stdout
#         old_stderr = sys.stderr
#         sys.stdout = devnull
#         sys.stderr = devnull
#         try:
#             yield
#         finally:
#             sys.stdout = old_stdout
#             sys.stderr = old_stderr

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg16(weights=None).features
        
    def forward(self, x):
        x = self.vgg(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(TransformerEncoder, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer(x)
        return x
    
class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim=512, output_dim=256):
        super(ProjectionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(AudioEncoder, self).__init__()
        self.feature_extractor = VGGFeatureExtractor()
        self.projection_network = ProjectionNetwork(input_dim=512, output_dim=embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.positional_encoding = nn.Parameter(torch.randn(1, 49, embed_dim))

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.feature_extractor(x)
        seq_len = x.size(2) * x.size(3)
        x = x.view(batch_size, 512, seq_len).permute(0, 2, 1)
        x = self.projection_network(x)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return x

class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(VideoEncoder, self).__init__()
        self.feature_extractor = VGGFeatureExtractor()
        self.projection_network = ProjectionNetwork(input_dim=512, output_dim=embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.embed_dim=embed_dim
        self.positional_encoding = nn.Parameter(torch.randn(1, 4, embed_dim))

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        x = self.feature_extractor(x)
        features = x.view(batch_size * num_frames, 512, -1).permute(0, 2, 1)
        features = self.projection_network(features)
        features = features + self.positional_encoding
        features = features.view(batch_size, num_frames, -1, self.embed_dim)
        features = features.mean(dim=2)
        features = self.transformer_encoder(features)
        features = features.mean(dim=1)
        return features

class TextEncoder(nn.Module):
    def __init__(self, model_name='roberta-base', embed_dim=256):
        super(TextEncoder, self).__init__()
        self.text_model = RobertaModel.from_pretrained(model_name)
        self.projection_network = ProjectionNetwork(self.text_model.config.hidden_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Freeze some layers
        for param in self.text_model.encoder.layer[:6]:
            for p in param.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        x = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.projection_network(x)
        x = self.norm(x)
        return x

class TransformerFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(TransformerFusion, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, 3, embed_dim))
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

    def forward(self, modalities_features):
        num_modalities = len(modalities_features)
        if num_modalities == 1:
            features = modalities_features[0].unsqueeze(1)
        else:
            features = torch.stack(modalities_features, dim=1)
        positional_encoding = self.positional_encoding[:, :num_modalities, :]
        features += positional_encoding
        x = self.transformer(features)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=4, num_layers=2, dropout=0.1): # , temperature=2.0):
        super(TransformerDecoder, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = embed_dim
        # self.temperature = temperature
        self.emotion_queries = nn.Parameter(torch.randn(num_classes, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, fused_features):
        batch_size = fused_features.size(0)
        emotion_queries = repeat(self.emotion_queries, 'num_classes embed_dim -> batch_size num_classes embed_dim', batch_size=batch_size)
        emotion_representations = self.transformer_decoder(
            tgt=emotion_queries,
            memory=fused_features
        )
        emotion_logits = self.fc_out(emotion_representations).squeeze(-1)
        # emotion_logits = emotion_logits / self.temperature
        emotion_probs = torch.softmax(emotion_logits, dim=1)
        return emotion_probs

class MultimodalEmotionRecognition(nn.Module):
    def __init__(self, enabled_modalities=["video", "audio", "text"], embed_dim=256, num_heads=4, num_layers=2, num_classes=6):
        super(MultimodalEmotionRecognition, self).__init__()
        self.enabled_modalities = enabled_modalities
        self.video_encoder = VideoEncoder(embed_dim=embed_dim) if "video" in enabled_modalities else None
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim) if "audio" in enabled_modalities else None
        self.text_encoder = TextEncoder(embed_dim=embed_dim) if "text" in enabled_modalities else None
        self.fusion = TransformerFusion(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.decoder = TransformerDecoder(embed_dim=embed_dim, num_classes=num_classes, num_heads=num_heads, num_layers=num_layers)

    def forward(self, video, audio, text_input_ids, text_attention_mask):
        embeddings = []
        if self.video_encoder and video is not None:
            embeddings.append(self.video_encoder(video))
        if self.audio_encoder and audio is not None:
            embeddings.append(self.audio_encoder(audio))
        if self.text_encoder and text_input_ids is not None and text_attention_mask is not None:
            embeddings.append(self.text_encoder(text_input_ids, text_attention_mask))
        fused_features = self.fusion(embeddings)
        predictions = self.decoder(fused_features)
        return predictions

class EmotionRecognizer:
    def __init__(self, model_path=MODEL_CHECKPOINT):
        # Nuclear suppression during initialization
        # with suppress_all():
            # Configure ONNX Runtime to be completely silent
            from onnxruntime import SessionOptions
            so = SessionOptions()
            so.log_severity_level = 4  # FATAL level (most severe)
            
            # Initialize FaceAnalysis with maximum suppression
            self.face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                session_options=so
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Load other components
            self.device = DEVICE
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.model = MultimodalEmotionRecognition(num_classes=NUM_CLASSES).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        
    def extract_audio_from_video(self, video_path, output_dir="temp_audio"):
        if output_dir is None:
            output_dir = os.path.join(get_base_path(), "temp_audio")
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = os.path.basename(video_path).split('.')[0]
        audio_path = os.path.join(output_dir, f"{video_name}.wav")
        
        if not os.path.exists(audio_path):
            # Method 1: Try using librosa first
            try:
                print(f"Attempting audio extraction with librosa from {video_path}")
                y, sr = librosa.load(video_path, sr=16000, mono=True)
                import soundfile as sf
                sf.write(audio_path, y, sr)
                print(f"Successfully extracted audio with librosa to {audio_path}")
                return audio_path
            except Exception as e:
                print(f"Librosa extraction failed: {str(e)}")
                
            # Method 2: If librosa fails, try PyAV
            try:
                import av
                
                print(f"Extracting audio using PyAV from {video_path}")
                input_container = av.open(video_path)
                
                # Find the first audio stream
                audio_stream = next((s for s in input_container.streams if s.type == 'audio'), None)
                
                if audio_stream is None:
                    raise RuntimeError("No audio stream found in the video file")
                
                # Create resampler
                resampler = av.audio.resampler.AudioResampler(
                    format='s16', 
                    layout='mono', 
                    rate=16000
                )
                
                # Open output file
                output_container = av.open(audio_path, mode='w')
                output_stream = output_container.add_stream('pcm_s16le', rate=16000)
                
                # Decode audio frames and encode to output
                for frame in input_container.decode(audio_stream):
                    # Resample if needed
                    if frame.sample_rate != 16000 or frame.layout.name != 'mono':
                        frames = resampler.resample(frame)
                    else:
                        frames = [frame]
                    
                    # Encode and mux
                    for frame in frames:
                        for packet in output_stream.encode(frame):
                            output_container.mux(packet)
                
                # Flush encoder
                for packet in output_stream.encode(None):
                    output_container.mux(packet)
                
                # Close files
                output_container.close()
                input_container.close()
                
                print(f"Successfully extracted audio with PyAV to {audio_path}")
                return audio_path
                
            except Exception as e:
                print(f"PyAV audio extraction failed: {str(e)}")
                
            # Method 3: If both methods fail, create a dummy audio file
            try:
                import numpy as np
                import soundfile as sf
                
                # Create 1 second of silence at 16kHz
                dummy_audio = np.zeros(16000, dtype=np.float32)
                sf.write(audio_path, dummy_audio, 16000)
                print(f"Created dummy silent audio file")
                return audio_path
                
            except Exception as dummy_error:
                raise RuntimeError(f"All audio extraction methods failed: {str(dummy_error)}")
        
        return audio_path
    
    def extract_mel_spectrogram(self, video_path):
        audio_path = self.extract_audio_from_video(video_path)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Check if audio is silent (all zeros or very low amplitude)
        if np.max(np.abs(y)) < 1e-6:
            print("Warning: Audio is silent or near-silent, using synthetic spectrogram")
            # Create a synthetic non-zero spectrogram instead of all zeros
            mel_normalized = np.random.uniform(0.1, 0.2, (224, 224))
        else:
            # Generate mel spectrogram from audio
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            mel_resized = cv2.resize(mel_spec_db, (224, 224), interpolation=cv2.INTER_CUBIC)
            
            # Safe normalization to avoid division by zero
            min_val = mel_resized.min()
            max_val = mel_resized.max()
            range_val = max_val - min_val
            
            # Check if range is too small (avoiding division by zero or near-zero)
            if range_val < 1e-6:
                print("Warning: Constant mel spectrogram detected, using synthetic values")
                mel_normalized = np.random.uniform(0.1, 0.2, mel_resized.shape)
            else:
                mel_normalized = (mel_resized - min_val) / range_val
        
        # Create RGB version of the spectrogram
        mel_rgb = np.stack([mel_normalized] * 3, axis=-1)
        
        # Convert to tensor and normalize
        mel_tensor = torch.tensor(mel_rgb, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        mel_normalized_tensor = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(mel_tensor)
        
        return mel_normalized_tensor
    
    def extract_video_frames(self, video_path, num_frames=10):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_idx, frame_rgb))
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                frames.append((frame_idx, placeholder))
        
        cap.release()
        return frames
    
    def process_faces(self, frames):
        faces_batch = []
        for frame_idx, frame_rgb in frames:
            face_dets = self.face_app.get(frame_rgb)
            for _, face in enumerate(face_dets):
                if face.bbox is None:
                    continue
                
                x1, y1, x2, y2 = map(int, face.bbox)
                x1 = max(0, min(x1, frame_rgb.shape[1]))
                y1 = max(0, min(y1, frame_rgb.shape[0]))
                x2 = max(0, min(x2, frame_rgb.shape[1]))
                y2 = max(0, min(y2, frame_rgb.shape[0]))

                if x2 <= x1 or y2 <= y1:
                    continue

                face_cropped = frame_rgb[y1:y2, x1:x2]
                if face_cropped.size == 0:
                    continue

                face_resized = cv2.resize(face_cropped, (64, 64))
                face_tensor = T.ToTensor()(face_resized)
                face_tensor_normalized = T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(face_tensor)
                faces_batch.append(face_tensor_normalized)
                
        if not faces_batch:
            placeholder = torch.zeros((3, 64, 64))
            faces_batch.append(placeholder)
                
        return faces_batch
    
    def process_text(self, text):
        encoding = self.tokenizer(
            text,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoding["input_ids"], encoding["attention_mask"]
    
    def predict_emotion(self, video_path, text=""):
        # with suppress_all():
            try:
                # Process video
                frames = self.extract_video_frames(video_path)
                faces = self.process_faces(frames)
                
                if len(faces) > 0:
                    video_tensor = torch.stack(faces)
                else:
                    video_tensor = torch.zeros((1, 3, 64, 64))
                
                num_faces = 4
                if len(video_tensor) > num_faces:
                    indices = torch.randperm(len(video_tensor))[:num_faces]
                    video_tensor = video_tensor[indices]
                elif len(video_tensor) < num_faces:
                    padding = torch.zeros((num_faces - len(video_tensor), 3, 64, 64))
                    video_tensor = torch.cat([video_tensor, padding])
                
                # Process audio
                mel_spec = self.extract_mel_spectrogram(video_path)
                
                # Process text
                input_ids, attention_mask = self.process_text(text)
                
                # Prepare inputs
                video_input = video_tensor.unsqueeze(0).to(self.device)
                audio_input = mel_spec.unsqueeze(0).to(self.device)
                text_input_ids = input_ids.unsqueeze(0).to(self.device)
                text_attention_mask = attention_mask.unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(
                        video_input, 
                        audio_input, 
                        text_input_ids, 
                        text_attention_mask
                    )
                
                # Get predictions
                probabilities = outputs.squeeze().cpu().numpy()
                predicted_class = np.argmax(probabilities)
                emotion = EMOTION_LABELS[predicted_class]
                
                # Format results
                results = {
                    "emotion": emotion,
                    "confidence": float(probabilities[predicted_class]),
                    "probabilities": {label: float(prob) for label, prob in zip(EMOTION_LABELS, probabilities)},
                    "timestamp": datetime.now().isoformat(),
                    "video_path": video_path
                }
                
                return results
                
            except Exception as e:
                return {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

def main(video_path, text=''):
    # Initialize recognizer with suppressed output
    # with suppress_all():
    recognizer = EmotionRecognizer()
    
    # Get video path from command line
    # video_path = sys.argv[1] if len(sys.argv) > 1 else None
    # text = sys.argv[2] if len(sys.argv) > 2 else ""
    
    if not video_path:
        print(json.dumps({"error": "No video path provided"}, indent=2))
        return
    
    result = recognizer.predict_emotion(video_path, text)
    return result

if __name__ == "__main__":
    # Example usage with your specific video path
    video_path = r"C:\Users\sarse\Downloads\GMT20250424-140111_Clip_Damir Sarsengaliyev's Clip 04_24_2025.mp4"
    if os.path.exists(video_path):
        print("_______CHECKING_________")
    sample_text = ""  # Optional text input
    
    try:
        # Initialize recognizer (with suppressed output during init)
        # with suppress_all():
        recognizer = EmotionRecognizer()
        
        # Process the video
        print(f"\nProcessing video: {video_path}...")
        result = recognizer.predict_emotion(video_path, sample_text)
        
        # Pretty-print results
        print("\nEmotion Recognition Results:")
        print("-" * 50)
        print('________',result)
        print(f"Predicted Emotion: {result['emotion']} (Confidence: {result['confidence']:.2%})")
        print("\nDetailed Probabilities:")
        for emotion, prob in result['probabilities'].items():
            print(f"{emotion.capitalize():<10}: {prob:.2%}")
        print("-" * 50)
        print(f"Analysis completed at: {result['timestamp']}")
        
    except Exception as e:
        print(f"\nError processing video:")
        print("-" * 50)
        print(f"File: {video_path}")
        print(f"Error: {str(e)}")
        print("-" * 50)
        print("Note: Ensure the file exists and is a valid video format")