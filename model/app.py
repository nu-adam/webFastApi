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
import whisper

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion labels (update based on your training)
EMOTION_LABELS = [
    "anger", "happiness", "neutral", 
    "sadness"
]

# Suppressing Logging:
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
        # print('Audio Encoder x1', x)
        batch_size, _, _, _ = x.shape
        x = self.feature_extractor(x)
        # print('Audio Encoder x2', x)
        seq_len = x.size(2) * x.size(3)
        # print('Audio Encoder seq len', seq_len)
        x = x.view(batch_size, 512, seq_len).permute(0, 2, 1)
        # print('Audio Encoder x3', x)
        x = self.projection_network(x)
        # print('Audio Encoder x4', x)
        x = x + self.positional_encoding
        # print('Audio Encoder x5', x)
        x = self.transformer_encoder(x)
        # print('Audio Encoder x6', x)
        x = x.mean(dim=1)
        # print('Audio Encoder 76', x)
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
        # print('Video Encoder x1', x)
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        # print('Video Encoder x2', x)
        x = self.feature_extractor(x)
        # print('Video Encoder x3', x)
        features = x.view(batch_size * num_frames, 512, -1).permute(0, 2, 1)
        # print('Video Encoder f1', x)
        features = self.projection_network(features)
        # print('Video Encoder f2', x)
        features = features + self.positional_encoding
        # print('Video Encoder f3', x)
        features = features.view(batch_size, num_frames, -1, self.embed_dim)
        # print('Video Encoder f4', x)
        features = features.mean(dim=2)
        # print('Video Encoder f5', x)
        features = self.transformer_encoder(features)
        # print('Video Encoder f6', x)
        features = features.mean(dim=1)
        # print('Video Encoder f7', x)
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
        # print('Text Encoder inp_ids1', input_ids)
        # print('Text Encoder att_msk1', attention_mask)
        input_ids = input_ids.squeeze(1)
        # print('Text Encoder inp_ids2', input_ids)
        attention_mask = attention_mask.squeeze(1)
        # print('Text Encoder att_msk2', attention_mask)
        x = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # print('Text Encoder x1', x)
        x = x.last_hidden_state[:, 0, :]
        # print('Text Encoder x2', x)
        x = self.projection_network(x)
        # print('Text Encoder x3', x)
        x = self.norm(x)
        # print('Text Encoder x4', x)
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
            # print('Fuser f1', features)
        positional_encoding = self.positional_encoding[:, :num_modalities, :]
        # print('Fuser pos_enc', positional_encoding)
        features += positional_encoding
        # print('Fuser f2', features)
        x = self.transformer(features)
        # print('Fuser x', x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=4, num_layers=2, dropout=0.1, temperature=2.0):
        super(TransformerDecoder, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = embed_dim
        self.temperature = temperature
        self.emotion_queries = nn.Parameter(torch.randn(num_classes, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, fused_features):
        # print('Decoder f1', fused_features)
        batch_size = fused_features.size(0)
        emotion_queries = repeat(self.emotion_queries, 'num_classes embed_dim -> batch_size num_classes embed_dim', batch_size=batch_size)
        # print('Decoder em_qrs', emotion_queries)
        emotion_representations = self.transformer_decoder(
            tgt=emotion_queries,
            memory=fused_features
        )
        # print('Decoder em_repr', emotion_representations)
        emotion_logits = self.fc_out(emotion_representations).squeeze(-1)
        # print('Decoder em_log', emotion_logits)
        emotion_logits = emotion_logits / self.temperature
        emotion_probs = torch.softmax(emotion_logits, dim=1)
        # print('Decoder em_prob', emotion_probs)
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
        
        # Initialize Whisper model
        self.whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
        # Move Whisper to device if using GPU/MPS
        if self.device.type in ["cuda", "mps"]:
            self.whisper_model.to(self.device)
            
    def extract_text_from_audio(self, audio_path):
        """
        Extract text from an audio file using Whisper.
        """
        try:
            # Load and transcribe audio
            result = self.whisper_model.transcribe(audio_path, language="en")  # Specify language if known
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return ""  # Return empty string on failure
    
    def extract_mel_spectrogram(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            if len(y) == 0 or np.all(y == 0):  # Check for empty or silent audio
                print("Warning: Audio is empty or silent. Returning zero tensor.")
                return torch.zeros((3, 224, 224), dtype=torch.float32, device=self.device)

            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

            # Check for NaNs or Infs in mel_spec_db
            if np.any(np.isnan(mel_spec_db)) or np.any(np.isinf(mel_spec_db)):
                print("Warning: Mel spectrogram contains NaNs or Infs. Returning zero tensor.")
                return torch.zeros((3, 224, 224), dtype=torch.float32, device=self.device)

            mel_resized = cv2.resize(mel_spec_db, (224, 224), interpolation=cv2.INTER_CUBIC)

            # Normalize mel_resized
            min_val = mel_resized.min()
            max_val = mel_resized.max()
            if max_val == min_val:  # Handle flat spectrogram
                print("Warning: Mel spectrogram is flat (max == min). Returning zero tensor.")
                return torch.zeros((3, 224, 224), dtype=torch.float32, device=self.device)

            mel_resized = (mel_resized - min_val) / (max_val - min_val)
            mel_rgb = np.stack([mel_resized] * 3, axis=-1)

            mel_tensor = torch.tensor(mel_rgb, dtype=torch.float32, device=self.device).permute(2, 0, 1)
            mel_normalized = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(mel_tensor)
            return mel_normalized

        except Exception as e:
            print(f"Error processing mel spectrogram for {video_path}: {str(e)}")
            return torch.zeros((3, 224, 224), dtype=torch.float32, device=self.device)
    
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
    
    def predict_emotion(self, video_path, audio_path, text):
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
            mel_spec = self.extract_mel_spectrogram(audio_path)
            
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
                "video_path": video_path,
                "transcribed_text": text  # Include transcribed text in results
            }
            
            return results
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

def main(video_path, audio_path, text):
    # Initialize recognizer with suppressed output
    # with suppress_all():
    recognizer = EmotionRecognizer()
    
    # Get video path from command line
    # video_path = sys.argv[1] if len(sys.argv) > 1 else None
    # text = sys.argv[2] if len(sys.argv) > 2 else ""
    
    if not video_path:
        print(json.dumps({"error": "No video path provided"}, indent=2))
        return
    
    result = recognizer.predict_emotion(video_path, audio_path, text)
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    # Example usage with your specific video path
    # video_path = "/Users/alikhanbaidussenov/Desktop/coding/projects/nu-adam/webFastApi/splits/videoplayback/videoplayback_part15.mp4"
    video_path = '/Users/alikhanbaidussenov/Desktop/coding/projects/nu-adam/webFastApi/video_splits/videoplayback/videoplayback_part1.mp4'
    audio_path = '/Users/alikhanbaidussenov/Desktop/coding/projects/nu-adam/webFastApi/audio_splits/videoplayback/videoplayback_part1.wav'
    
    try:
        # Initialize recognizer (with suppressed output during init)
        # with suppress_all():
        recognizer = EmotionRecognizer()
        
        # Process the video
        print(f"\nProcessing video: {video_path}...")
        result = recognizer.predict_emotion(video_path, audio_path,'My parents, but things are not being good too early.')
        
        # Pretty-print results
        print("\nEmotion Recognition Results:")
        print("-" * 50)
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