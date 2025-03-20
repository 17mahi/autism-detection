#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flask application for Autism Detection and Emotion Analysis.
This file contains the main routing logic and API endpoints.
"""

import os
import logging
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import numpy as np
import cv2
from werkzeug.utils import secure_filename

# Import our modules
from src.models.emotion_detector import EmotionDetector
from src.models.autism_detector import AutismDetector
from src.utils.data_processor import DataProcessor
from src.visualization.visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__,
            static_folder='static',
            template_folder='static/templates')
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_development')

# Set up upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

# Initialize our models
try:
    emotion_detector = EmotionDetector()
    autism_detector = AutismDetector()
    data_processor = DataProcessor()
    visualizer = Visualizer()
    logger.info("Models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    # Continue without models for development purposes
    emotion_detector = None
    autism_detector = None

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/screening')
def screening():
    """Render the autism screening page."""
    return render_template('screening.html')

@app.route('/emotion')
def emotion():
    """Render the emotion analysis page."""
    return render_template('emotion.html')

@app.route('/api/v1/analyze/emotion', methods=['POST'])
def analyze_emotion():
    """API endpoint for analyzing emotions in an image."""
    if 'media' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        }), 400
    
    file = request.files['media']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400
    
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({
            'status': 'error',
            'message': 'File type not allowed. Please upload an image (png, jpg, jpeg, gif).'
        }), 400
    
    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(file_path)
        
        # Process the image with emotion detector
        if emotion_detector:
            # Read the image
            image = cv2.imread(file_path)
            
            # Analyze emotions
            result = emotion_detector.detect_emotion(image)
            
            # Draw emotion on image
            processed_image = emotion_detector.draw_emotion(image, result)
            
            # Convert processed image to base64 for response
            _, buffer = cv2.imencode('.jpg', processed_image)
            processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare response
            response = {
                'status': 'success',
                'result': {
                    'primary_emotion': result['primary_emotion'],
                    'emotion_scores': result['emotion_scores'],
                    'faces_detected': result['faces_detected'],
                    'processed_image': processed_image_b64
                }
            }
        else:
            # Mock response for development
            logger.warning("Using mock emotion detection response (no model loaded)")
            response = {
                'status': 'success',
                'result': {
                    'primary_emotion': 'happy',
                    'emotion_scores': {
                        'happy': 0.8,
                        'sad': 0.05,
                        'angry': 0.02,
                        'surprise': 0.03,
                        'fear': 0.01,
                        'disgust': 0.02,
                        'neutral': 0.07
                    },
                    'faces_detected': 1,
                    'processed_image': '' # Would contain base64 encoded image in production
                }
            }
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error analyzing emotion: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({
            'status': 'error',
            'message': f'Error processing image: {str(e)}'
        }), 500

@app.route('/api/v1/analyze/video', methods=['POST'])
def analyze_video():
    """API endpoint for analyzing emotions in a video."""
    if 'media' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        }), 400
    
    file = request.files['media']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400
    
    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({
            'status': 'error',
            'message': 'File type not allowed. Please upload a video (mp4, avi, mov, wmv).'
        }), 400
    
    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(file_path)
        
        # Process the video with emotion detector
        if emotion_detector:
            # Analyze video for emotions
            result = emotion_detector.analyze_video(file_path)
            
            # Prepare response
            response = {
                'status': 'success',
                'result': result
            }
        else:
            # Mock response for development
            logger.warning("Using mock video analysis response (no model loaded)")
            response = {
                'status': 'success',
                'result': {
                    'predominant_emotion': 'happy',
                    'emotion_distribution': {
                        'happy': 0.65,
                        'sad': 0.05,
                        'angry': 0.03,
                        'surprise': 0.12,
                        'fear': 0.02,
                        'disgust': 0.03,
                        'neutral': 0.1
                    },
                    'total_frames': 120,
                    'total_faces_detected': 87,
                    'processing_time': 5.2
                }
            }
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({
            'status': 'error',
            'message': f'Error processing video: {str(e)}'
        }), 500

@app.route('/api/v1/screening/submit', methods=['POST'])
def submit_screening():
    """API endpoint for submitting autism screening questionnaire."""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Process screening data with autism detector
        if autism_detector:
            # Analyze screening answers
            result = autism_detector.analyze_screening(data)
            
            # Save to session for potential later reference
            session['screening_result'] = result
            
            # Prepare response
            response = {
                'status': 'success',
                'result': result
            }
        else:
            # Mock response for development
            logger.warning("Using mock autism screening response (no model loaded)")
            response = {
                'status': 'success',
                'result': {
                    'risk_score': 0.35,
                    'risk_level': 'low',
                    'feature_importance': {
                        'eye_contact': 0.3,
                        'social_interaction': 0.25,
                        'repetitive_behavior': 0.2,
                        'verbal_communication': 0.15,
                        'nonverbal_communication': 0.1
                    },
                    'recommendations': [
                        'Consider a follow-up with a pediatrician',
                        'Monitor social development',
                        'Engage in interactive play',
                        'Read books together daily',
                        'Practice joint attention activities'
                    ]
                }
            }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing screening: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error processing screening: {str(e)}'
        }), 500

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000) 