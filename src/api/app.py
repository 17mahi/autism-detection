#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flask application for the Autism Detection and Emotion Analysis System.
This module defines the API endpoints and web interface for the application.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_wtf.csrf import CSRFProtect

# Set up logging
logger = logging.getLogger(__name__)

def create_app(test_config=None):
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        static_folder="../../static",
        template_folder="../../static/templates"
    )
    
    # Configure app
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev_key_for_development_only'),
        UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../static/uploads'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16 MB max upload
    )
    
    # Ensure upload directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Initialize CSRF protection
    csrf = CSRFProtect(app)
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../static/templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        # Create a basic index.html template
        with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Autism Detection and Emotion Analysis</title>
                <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body>
                <header>
                    <h1>AI-Powered Autism Detection and Emotion Analysis</h1>
                </header>
                <main>
                    <section class="intro">
                        <h2>Welcome to the Autism Detection and Emotion Analysis System</h2>
                        <p>This tool uses artificial intelligence to assist in the early detection of autism spectrum disorder (ASD) and analyze emotional responses in children.</p>
                    </section>
                    <section class="features">
                        <div class="feature-card">
                            <h3>Autism Screening</h3>
                            <p>Upload videos or answer questionnaires to help identify behavioral patterns associated with autism.</p>
                            <a href="/screening" class="button">Start Screening</a>
                        </div>
                        <div class="feature-card">
                            <h3>Emotion Analysis</h3>
                            <p>Analyze facial expressions and vocal patterns to identify emotional states.</p>
                            <a href="/emotion" class="button">Analyze Emotions</a>
                        </div>
                        <div class="feature-card">
                            <h3>Progress Tracking</h3>
                            <p>Monitor changes in emotional and behavioral patterns over time.</p>
                            <a href="/tracking" class="button">Track Progress</a>
                        </div>
                    </section>
                </main>
                <footer>
                    <p>&copy; 2023 AI Autism Detection Project</p>
                    <p><small>Disclaimer: This tool is not a medical device and should not be used for diagnosing autism spectrum disorder. Always consult with healthcare professionals.</small></p>
                </footer>
                <script src="{{ url_for('static', filename='js/main.js') }}"></script>
            </body>
            </html>
            """)
    
    # Create CSS directory and file if they don't exist
    css_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../static/css')
    if not os.path.exists(css_dir):
        os.makedirs(css_dir)
        # Create a basic CSS file
        with open(os.path.join(css_dir, 'main.css'), 'w') as f:
            f.write("""
            /* Main CSS for Autism Detection and Emotion Analysis System */
            
            :root {
                --primary-color: #4a90e2;
                --secondary-color: #5c6bc0;
                --accent-color: #7e57c2;
                --text-color: #333;
                --light-bg: #f5f7fa;
                --dark-bg: #263238;
                --success-color: #66bb6a;
                --warning-color: #ffa726;
                --error-color: #ef5350;
            }
            
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--light-bg);
            }
            
            header {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                padding: 2rem 1rem;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            main {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem 1rem;
            }
            
            h1, h2, h3 {
                margin-bottom: 1rem;
            }
            
            .intro {
                text-align: center;
                margin-bottom: 3rem;
            }
            
            .features {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 2rem;
            }
            
            .feature-card {
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 2rem;
                flex: 1 1 300px;
                max-width: 350px;
                transition: transform 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
            }
            
            .button {
                display: inline-block;
                background-color: var(--primary-color);
                color: white;
                padding: 0.75rem 1.5rem;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 1rem;
                transition: background-color 0.3s ease;
            }
            
            .button:hover {
                background-color: var(--secondary-color);
            }
            
            footer {
                background-color: var(--dark-bg);
                color: white;
                text-align: center;
                padding: 2rem;
                margin-top: 3rem;
            }
            
            footer small {
                display: block;
                margin-top: 1rem;
                opacity: 0.8;
            }
            
            @media (max-width: 768px) {
                .features {
                    flex-direction: column;
                    align-items: center;
                }
                
                .feature-card {
                    max-width: 100%;
                }
            }
            """)
    
    # Create JS directory and file if they don't exist
    js_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../static/js')
    if not os.path.exists(js_dir):
        os.makedirs(js_dir)
        # Create a basic JS file
        with open(os.path.join(js_dir, 'main.js'), 'w') as f:
            f.write("""
            // Main JavaScript for Autism Detection and Emotion Analysis System
            
            document.addEventListener('DOMContentLoaded', function() {
                console.log('Autism Detection and Emotion Analysis System loaded');
                
                // Add smooth scrolling
                document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                    anchor.addEventListener('click', function (e) {
                        e.preventDefault();
                        
                        document.querySelector(this.getAttribute('href')).scrollIntoView({
                            behavior: 'smooth'
                        });
                    });
                });
            });
            """)

    # Register routes
    @app.route('/')
    def home():
        """Render the home page."""
        return render_template('index.html')
    
    @app.route('/screening')
    def screening():
        """Render the autism screening page."""
        return render_template('screening.html')
    
    @app.route('/emotion')
    def emotion():
        """Render the emotion analysis page."""
        return render_template('emotion.html')
    
    @app.route('/tracking')
    def tracking():
        """Render the progress tracking page."""
        return render_template('tracking.html')
    
    # API routes
    @app.route('/api/v1/analyze/video', methods=['POST'])
    def analyze_video():
        """API endpoint for analyzing video for autism markers."""
        # This would call the model prediction functions
        return jsonify({
            "status": "success",
            "message": "Video analysis complete",
            "result": {
                "autism_indicators": {
                    "score": 0.65,
                    "confidence": 0.78,
                    "markers_detected": ["limited eye contact", "repetitive movements"]
                }
            }
        })
    
    @app.route('/api/v1/analyze/emotion', methods=['POST'])
    def analyze_emotion():
        """API endpoint for analyzing emotions in image or video."""
        # This would call the emotion detection model
        return jsonify({
            "status": "success",
            "message": "Emotion analysis complete",
            "result": {
                "primary_emotion": "happy",
                "emotion_scores": {
                    "happy": 0.82,
                    "sad": 0.05,
                    "angry": 0.03,
                    "surprised": 0.07,
                    "neutral": 0.03
                }
            }
        })
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        return render_template('500.html'), 500
    
    # Register more blueprints here...
    
    return app 