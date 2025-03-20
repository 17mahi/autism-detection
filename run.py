#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entry point for the Autism Detection and Emotion Analysis System.
This script sets up the necessary environment and starts the Flask server.
"""

import os
import sys
from dotenv import load_dotenv
from src.app import create_app

# Load environment variables
load_dotenv()

def main():
    """
    Main function to run the application.
    
    This function configures the environment and runs the Flask application.
    """
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Create and run the application
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main() 