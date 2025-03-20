#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application file for the Autism Detection and Emotion Analysis System.
This module initializes the Flask application and configures the necessary settings.
"""

import os
import logging
from flask import Flask, render_template, send_from_directory
from dotenv import load_dotenv
from .api.routes import api

# Load environment variables from .env file if it exists
if os.path.exists(".env"):
    load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: Configured Flask application instance
    """
    # Initialize Flask app
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    
    # Configure app
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production'),
        UPLOAD_FOLDER=os.environ.get('UPLOAD_FOLDER', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')),
        MAX_CONTENT_LENGTH=int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB max file size by default
    )
    
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    
    # Register routes
    @app.route('/')
    def index():
        """Render the home page."""
        return render_template('index.html')
    
    @app.route('/screening')
    def screening():
        """Render the screening page."""
        return render_template('screening.html')
    
    @app.route('/emotion')
    def emotion():
        """Render the emotion analysis page."""
        return render_template('emotion.html')
    
    @app.route('/about')
    def about():
        """Render the about page."""
        return render_template('about.html')
    
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors."""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return render_template('500.html'), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False) 