from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    google_id = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    videos = db.relationship('Video', backref='user', lazy=True)

class Video(db.Model):
    __tablename__ = 'videos'
    
    video_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(255), nullable=False)
    graph_path = db.Column(db.String(255))
    happiness = db.Column(db.Float)
    frustration = db.Column(db.Float)
    anger = db.Column(db.Float)
    sadness = db.Column(db.Float)
    neutral = db.Column(db.Float)
    excited = db.Column(db.Float)
    transcription = db.Column(db.Text, default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)