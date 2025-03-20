#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Processor for the Autism Detection and Emotion Analysis System.
This module provides utilities for data preprocessing and transformation.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import cv2
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class for preprocessing and transforming data for autism detection and emotion analysis.
    """
    
    def __init__(self):
        """
        Initialize the data processor.
        """
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def preprocess_image(self, image_path, target_size=(224, 224), normalize=True, grayscale=False):
        """
        Preprocess an image for model input.
        
        Args:
            image_path (str): Path to the image file.
            target_size (tuple): Target image size (height, width).
            normalize (bool): Whether to normalize pixel values.
            grayscale (bool): Whether to convert to grayscale.
            
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            
            if img is None:
                logger.error(f"Error loading image: {image_path}")
                return None
            
            # Convert to grayscale if required
            if grayscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Add channel dimension for model
                img = np.expand_dims(img, axis=-1)
            
            # Resize image
            img = cv2.resize(img, target_size)
            
            # Normalize pixel values if required
            if normalize:
                img = img.astype(np.float32) / 255.0
            
            return img
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def batch_preprocess_images(self, image_paths, **kwargs):
        """
        Preprocess a batch of images.
        
        Args:
            image_paths (list): List of image file paths.
            **kwargs: Additional arguments for preprocess_image.
            
        Returns:
            numpy.ndarray: Batch of preprocessed images.
        """
        preprocessed_images = []
        
        for image_path in image_paths:
            img = self.preprocess_image(image_path, **kwargs)
            if img is not None:
                preprocessed_images.append(img)
        
        if not preprocessed_images:
            logger.error("No images were successfully preprocessed")
            return None
        
        return np.array(preprocessed_images)
    
    def preprocess_video(self, video_path, frame_interval=5, max_frames=30, **kwargs):
        """
        Preprocess a video by extracting and preprocessing frames.
        
        Args:
            video_path (str): Path to the video file.
            frame_interval (int): Interval between frames to extract.
            max_frames (int): Maximum number of frames to extract.
            **kwargs: Additional arguments for preprocess_image.
            
        Returns:
            numpy.ndarray: Preprocessed video frames.
        """
        try:
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Error opening video file: {video_path}")
                return None
            
            preprocessed_frames = []
            frame_count = 0
            
            # Process frames
            while len(preprocessed_frames) < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process every N frames
                if frame_count % frame_interval == 0:
                    # Convert frame to image
                    frame_path = os.path.join(self.data_dir, f"temp_frame_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Preprocess frame
                    preprocessed_frame = self.preprocess_image(frame_path, **kwargs)
                    
                    # Delete temporary frame file
                    os.remove(frame_path)
                    
                    if preprocessed_frame is not None:
                        preprocessed_frames.append(preprocessed_frame)
                
                frame_count += 1
            
            # Release video capture
            cap.release()
            
            if not preprocessed_frames:
                logger.error("No frames were successfully preprocessed")
                return None
            
            return np.array(preprocessed_frames)
        
        except Exception as e:
            logger.error(f"Error preprocessing video: {str(e)}")
            return None
    
    def normalize_questionnaire_data(self, data):
        """
        Normalize questionnaire data for model input.
        
        Args:
            data (dict): Questionnaire responses.
            
        Returns:
            dict: Normalized questionnaire data.
        """
        try:
            # Define response mappings
            response_mappings = {
                "never": 0,
                "rarely": 1,
                "sometimes": 2,
                "often": 3,
                "always": 4
            }
            
            # Normalize responses
            normalized_data = {}
            
            for question_id, response in data.items():
                if response in response_mappings:
                    normalized_data[question_id] = response_mappings[response] / 4.0  # Normalize to [0, 1]
                else:
                    # Try to convert to float if possible
                    try:
                        normalized_data[question_id] = float(response)
                        # If the value is outside [0, 1], normalize it
                        if normalized_data[question_id] < 0 or normalized_data[question_id] > 1:
                            normalized_data[question_id] = max(0, min(1, normalized_data[question_id]))
                    except ValueError:
                        logger.warning(f"Skipping invalid response for {question_id}: {response}")
            
            return normalized_data
        
        except Exception as e:
            logger.error(f"Error normalizing questionnaire data: {str(e)}")
            return {}
    
    def save_results(self, results, result_type, subject_id=None):
        """
        Save analysis results to a file.
        
        Args:
            results (dict): Analysis results.
            result_type (str): Type of results ('autism' or 'emotion').
            subject_id (str, optional): Subject identifier.
            
        Returns:
            str: Path to the saved results file.
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join(self.data_dir, 'results')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            subject_str = f"_{subject_id}" if subject_id else ""
            filename = f"{result_type}_results{subject_str}_{timestamp}.json"
            file_path = os.path.join(results_dir, filename)
            
            # Add metadata
            results_with_metadata = {
                "result_type": result_type,
                "timestamp": timestamp,
                "subject_id": subject_id,
                "results": results
            }
            
            # Save results
            with open(file_path, 'w') as f:
                json.dump(results_with_metadata, f, indent=4)
            
            logger.info(f"Results saved to {file_path}")
            
            return file_path
        
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return None
    
    def load_results(self, file_path):
        """
        Load analysis results from a file.
        
        Args:
            file_path (str): Path to the results file.
            
        Returns:
            dict: Loaded results.
        """
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Results loaded from {file_path}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return None

if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    
    # Test normalizing questionnaire data
    test_data = {
        "q1": "sometimes",
        "q2": "often",
        "q3": "rarely",
        "q4": "always",
        "q5": "never"
    }
    
    normalized_data = processor.normalize_questionnaire_data(test_data)
    
    print("Original Data:")
    print(test_data)
    print("\nNormalized Data:")
    print(normalized_data) 