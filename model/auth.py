# auth.py
from flask import Blueprint, request, jsonify, redirect, url_for, session, current_app
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from model.models import db, User
import json
from datetime import datetime
from functools import wraps

# IMPORTANT - Fix the way we access the client ID
GOOGLE_CLIENT_ID = "218236351163-0e7j1ctpkfnn7avk741nevbcfkscoa61.apps.googleusercontent.com"

auth_bp = Blueprint('auth', __name__)

login_manager = LoginManager()

def init_auth(app):
    login_manager.init_app(app)
    app.register_blueprint(auth_bp)
    
    # Set up the login manager
    login_manager.login_view = 'auth.login'
    
    # Print environment variable for debugging
    client_id_env = app.config.get('GOOGLE_CLIENT_ID', 'NOT SET')
    print("==== GOOGLE CLIENT ID CONFIG ====")
    print(f"From environment: {client_id_env}")
    print(f"Hardcoded value: {GOOGLE_CLIENT_ID}")
    print("=================================")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Add user_loader property to User model for Flask-Login
def get_id(self):
    return str(self.user_id)

User.is_authenticated = property(lambda self: True)
User.is_active = property(lambda self: True)
User.is_anonymous = property(lambda self: False)
User.get_id = get_id

# Authentication decorator for API routes
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            # Verify token with Google - using the hardcoded ID
            idinfo = id_token.verify_oauth2_token(
                token, 
                google_requests.Request(), 
                GOOGLE_CLIENT_ID  # Fixed: Use the hardcoded value
            )
            
            # Get user from database or create if not exists
            google_id = idinfo['sub']
            user = User.query.filter_by(google_id=google_id).first()
            
            if not user:
                return jsonify({'message': 'User not found!'}), 401
                
        except Exception as e:
            return jsonify({'message': 'Invalid token!', 'error': str(e)}), 401
            
        return f(user, *args, **kwargs)
    
    return decorated

@auth_bp.route('/auth/google', methods=['POST'])
def google_auth():
    # Get token from request
    token = request.json.get('token')
    
    print("==== RECEIVED AUTHENTICATION REQUEST ====")
    print(f"Token received: {token[:20]}..." if token else "No token")
    print("=========================================")
    
    if not token:
        return jsonify({'error': 'No token provided'}), 400
    
    try:
        print("Attempting to verify token with Google...")
        
        # Verify token with Google - using the hardcoded ID
        idinfo = id_token.verify_oauth2_token(
            token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID  # Fixed: Use the hardcoded value
        )
        
        print("Token verified successfully!")
        print(f"User email: {idinfo.get('email')}")
        print(f"User ID: {idinfo.get('sub')}")
        
        # Get user info
        google_id = idinfo['sub']
        email = idinfo['email']
        name = idinfo.get('name', email.split('@')[0])
        
        print(f"Looking for existing user with google_id: {google_id}")
        
        # Find user or create a new one
        user = User.query.filter_by(google_id=google_id).first()
        
        if not user:
            print(f"Creating new user with email: {email}")
            # Create new user
            user = User(
                email=email,
                name=name,
                google_id=google_id
            )
            db.session.add(user)
            db.session.commit()
            print(f"New user created with ID: {user.user_id}")
        else:
            print(f"Found existing user: {user.email}")
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
        
        # Login user with Flask-Login
        login_user(user)
        print("User logged in successfully")
        
        # Return user info and token
        return jsonify({
            'user': {
                'id': user.user_id,
                'name': user.name,
                'email': user.email
            },
            'token': token
        })
    
    except Exception as e:
        print(f"==== AUTHENTICATION ERROR ====")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"===============================")
        return jsonify({'error': str(e)}), 401

@auth_bp.route('/auth/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'})

@auth_bp.route('/auth/user', methods=['GET'])
@token_required
def get_user(user):
    return jsonify({
        'user': {
            'id': user.user_id,
            'name': user.name,
            'email': user.email
        }
    })