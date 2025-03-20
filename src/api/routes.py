#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API routes for the Autism Detection and Emotion Analysis System.
This module handles all API endpoints for the application.
"""

import os
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import numpy as np
from ..models.autism_detector import AutismDetector
from ..models.emotion_analyzer import EmotionAnalyzer
from ..utils.visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize models
autism_detector = AutismDetector()
emotion_analyzer = EmotionAnalyzer()
visualizer = Visualizer()

def allowed_file(filename, allowed_extensions):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@api.route('/v1/analyze/screening', methods=['POST'])
def analyze_screening():
    """
    Analyze screening questionnaire responses and optional video data.
    
    Returns:
        JSON response containing analysis results and recommendations.
    """
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert form values to float
        features = {k: float(v) for k, v in form_data.items() if k not in ['video', 'csrf_token']}
        
        # Process video if provided
        video_analysis = None
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file and video_file.filename and allowed_file(video_file.filename, {'mp4', 'avi', 'mov'}):
                filename = secure_filename(video_file.filename)
                upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
                os.makedirs(upload_folder, exist_ok=True)
                video_path = os.path.join(upload_folder, filename)
                video_file.save(video_path)
                
                try:
                    # Analyze video for emotion patterns
                    video_analysis = emotion_analyzer.analyze_video(video_path)
                finally:
                    # Clean up uploaded file
                    if os.path.exists(video_path):
                        os.remove(video_path)
        
        # Get autism screening prediction
        prediction = autism_detector.predict(features)
        
        # Prepare behavioral analysis
        behavioral_analysis = analyze_behavioral_patterns(features, prediction)
        
        # Generate recommendations
        recommendations = generate_recommendations(prediction, behavioral_analysis)
        
        # Prepare response
        response = {
            'status': 'success',
            'result': {
                'prediction_summary': {
                    'interpretation': prediction['interpretation'],
                    'confidence': prediction['confidence'],
                    'probability': prediction['probability']
                },
                'behavioral_analysis': behavioral_analysis,
                'recommendations': recommendations,
                'disclaimer': "This is a screening tool and not a diagnostic instrument. Results should be discussed with healthcare professionals."
            }
        }
        
        # Add video analysis if available
        if video_analysis:
            response['result']['video_analysis'] = video_analysis
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing screening request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Unable to process screening request. Please try again.'
        }), 500

@api.route('/v1/analyze/emotion', methods=['POST'])
def analyze_emotion():
    """
    Analyze emotion from an image or video.
    
    Returns:
        JSON response containing emotion analysis results.
    """
    try:
        if 'file' not in request.files and 'media' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        # Check both possible field names for the uploaded file
        media_file = request.files.get('file') or request.files.get('media')
        
        if not media_file or not media_file.filename:
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Check allowed file types
        allowed_image_extensions = {'jpg', 'jpeg', 'png'}
        allowed_video_extensions = {'mp4', 'avi', 'mov'}
        allowed_extensions = allowed_image_extensions.union(allowed_video_extensions)
        
        if not allowed_file(media_file.filename, allowed_extensions):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(media_file.filename)
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        media_file.save(file_path)
        
        try:
            # Analyze based on file type
            if filename.lower().endswith(tuple(allowed_image_extensions)):
                results = emotion_analyzer.analyze_image(file_path)
            else:
                results = emotion_analyzer.analyze_video(file_path)
            
            return jsonify({
                'status': 'success',
                'result': results
            })
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    except Exception as e:
        logger.error(f"Error processing emotion analysis request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Unable to process emotion analysis request. Please try again.'
        }), 500

def analyze_behavioral_patterns(features, prediction):
    """
    Analyze behavioral patterns from questionnaire responses.
    
    Args:
        features (dict): Questionnaire responses
        prediction (dict): Model prediction results
    
    Returns:
        dict: Behavioral analysis results
    """
    # Define thresholds for different behavioral patterns
    thresholds = {
        'social_interaction': 0.5,
        'communication': 0.5,
        'repetitive_behavior': 0.7,
        'sensory_sensitivity': 0.6
    }
    
    # Calculate scores for different behavioral areas
    social_keys = ['eye_contact', 'responds_to_name', 'social_smile', 'interest_in_peers']
    social_values = [features.get(k, 0) for k in social_keys if k in features]
    social_interaction_score = np.mean(social_values) if social_values else 0
    
    communication_keys = ['pointing', 'showing', 'pretend_play', 'follows_gaze']
    communication_values = [features.get(k, 0) for k in communication_keys if k in features]
    communication_score = np.mean(communication_values) if communication_values else 0
    
    repetitive_keys = ['repetitive_behaviors', 'unusual_interests']
    repetitive_values = [features.get(k, 0) for k in repetitive_keys if k in features]
    repetitive_behavior_score = np.mean(repetitive_values) if repetitive_values else 0
    
    # Identify areas of concern and strengths
    areas_of_concern = []
    strengths = []
    
    if social_interaction_score < thresholds['social_interaction']:
        areas_of_concern.append('social_interaction')
    else:
        strengths.append('social_interaction')
    
    if communication_score < thresholds['communication']:
        areas_of_concern.append('communication')
    else:
        strengths.append('communication')
    
    if repetitive_behavior_score > thresholds['repetitive_behavior']:
        areas_of_concern.append('repetitive_behavior')
    
    return {
        'areas_of_concern': areas_of_concern,
        'strengths': strengths,
        'scores': {
            'social_interaction': float(social_interaction_score),
            'communication': float(communication_score),
            'repetitive_behavior': float(repetitive_behavior_score)
        }
    }

def generate_recommendations(prediction, behavioral_analysis):
    """
    Generate recommendations based on screening results.
    
    Args:
        prediction (dict): Model prediction results
        behavioral_analysis (dict): Behavioral analysis results
    
    Returns:
        dict: Recommendations for follow-up actions
    """
    # Base recommendations
    recommendations = {
        'primary_recommendation': '',
        'suggested_interventions': []
    }
    
    # Determine primary recommendation based on prediction
    if prediction['probability'] > 0.7:
        recommendations['primary_recommendation'] = (
            "Based on the screening results, we recommend consulting with a healthcare "
            "professional for a comprehensive evaluation. The responses suggest a higher "
            "likelihood of autism spectrum traits."
        )
    elif prediction['probability'] > 0.4:
        recommendations['primary_recommendation'] = (
            "The screening results indicate some developmental concerns. We recommend "
            "monitoring these behaviors and consulting with a pediatrician or developmental "
            "specialist for further assessment."
        )
    else:
        recommendations['primary_recommendation'] = (
            "The screening results suggest typical development. However, if you have "
            "specific concerns about your child's development, we recommend discussing "
            "them with your pediatrician."
        )
    
    # Add specific interventions based on areas of concern
    interventions = {
        'social_interaction': [
            "Practice turn-taking games and activities",
            "Engage in joint attention activities",
            "Use social stories to explain social situations"
        ],
        'communication': [
            "Use visual supports and picture cards",
            "Practice imitation games",
            "Engage in pretend play activities"
        ],
        'repetitive_behavior': [
            "Introduce structured routines",
            "Use visual schedules",
            "Practice flexible thinking activities"
        ]
    }
    
    for area in behavioral_analysis['areas_of_concern']:
        if area in interventions:
            recommendations['suggested_interventions'].extend(interventions[area])
    
    return recommendations 