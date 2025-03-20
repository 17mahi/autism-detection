#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Emotion Detector Model for the Autism Detection and Emotion Analysis System.
This module processes images and videos to detect and recognize emotions in faces.
"""

import os
import logging
import time
import cv2
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionDetector:
    """
    A class for detecting and analyzing emotions in images and videos.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion detector with pre-trained models.
        
        Args:
            model_path (str, optional): Path to the emotion detection model.
                If None, default models will be used.
        """
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (130, 0, 75),   # Purple
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 255, 0),   # Green
            'neutral': (128, 128, 128) # Gray
        }
        
        # Load models
        try:
            # Load face detector
            self.face_cascade_path = os.path.join(
                os.path.dirname(__file__), 
                'haarcascade_frontalface_default.xml'
            )
            
            if os.path.exists(self.face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
                logger.info("Face detection model loaded successfully")
            else:
                # Use OpenCV's built-in cascades
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                logger.info("Using OpenCV's built-in face detection model")
            
            # In a real implementation, we would load a deep learning model for emotion recognition
            # For demonstration purposes, we'll generate random emotions
            logger.warning("Using mock emotion detector (random emotions)")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading emotion detection models: {str(e)}")
            self.model_loaded = False
    
    def preprocess_image(self, image):
        """
        Preprocess the image for emotion detection.
        
        Args:
            image (numpy.ndarray): Input image in BGR format.
            
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize if needed
        # gray = cv2.resize(gray, (48, 48))
        
        return gray
    
    def detect_faces(self, image):
        """
        Detect faces in the image.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            list: List of face rectangles (x, y, w, h).
        """
        if not self.model_loaded:
            logger.error("Face detection model not loaded")
            return []
        
        # Preprocess image
        gray = self.preprocess_image(image)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def detect_emotion(self, image):
        """
        Detect emotions in the image.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            dict: Dictionary containing emotion detection results.
        """
        if not self.model_loaded:
            logger.error("Emotion detection model not loaded")
            return {
                'primary_emotion': 'unknown',
                'emotion_scores': {},
                'faces_detected': 0
            }
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            logger.info("No faces detected in the image")
            return {
                'primary_emotion': 'unknown',
                'emotion_scores': {},
                'faces_detected': 0
            }
        
        # For demonstration, we'll generate random emotions
        # In a real implementation, we would use a deep learning model to predict emotions
        
        # Simulate emotion prediction
        emotion_idx = np.random.choice(len(self.emotions), p=[0.1, 0.05, 0.05, 0.4, 0.1, 0.1, 0.2])
        primary_emotion = self.emotions[emotion_idx]
        
        # Generate random emotion scores that sum to 1
        emotion_scores = np.random.dirichlet(np.ones(len(self.emotions)))
        emotion_scores_dict = {emotion: float(score) for emotion, score in zip(self.emotions, emotion_scores)}
        
        # Ensure the primary emotion has the highest score
        max_score = max(emotion_scores_dict.values())
        for emotion in emotion_scores_dict:
            if emotion == primary_emotion:
                emotion_scores_dict[emotion] = max_score
            else:
                emotion_scores_dict[emotion] = min(emotion_scores_dict[emotion], max_score - 0.1)
        
        # Normalize scores to sum to 1
        total = sum(emotion_scores_dict.values())
        emotion_scores_dict = {emotion: score/total for emotion, score in emotion_scores_dict.items()}
        
        result = {
            'primary_emotion': primary_emotion,
            'emotion_scores': emotion_scores_dict,
            'faces_detected': len(faces)
        }
        
        return result
    
    def draw_emotion(self, image, result):
        """
        Draw emotion detection results on the image.
        
        Args:
            image (numpy.ndarray): Input image.
            result (dict): Emotion detection results.
            
        Returns:
            numpy.ndarray: Image with emotion annotations.
        """
        if not self.model_loaded or 'primary_emotion' not in result or result['primary_emotion'] == 'unknown':
            return image
        
        # Create a copy of the image
        img_copy = image.copy()
        
        # Detect faces again to get coordinates
        faces = self.detect_faces(image)
        
        # Get the primary emotion and its color
        primary_emotion = result['primary_emotion']
        color = self.emotion_colors.get(primary_emotion, (255, 255, 255))
        
        # Draw on each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
            
            # Add emotion label
            label = f"{primary_emotion}: {result['emotion_scores'][primary_emotion]:.2f}"
            cv2.putText(img_copy, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return img_copy
    
    def analyze_video(self, video_path):
        """
        Analyze emotions in a video.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            dict: Dictionary containing video emotion analysis results.
        """
        if not self.model_loaded:
            logger.error("Emotion detection model not loaded")
            return {
                'predominant_emotion': 'unknown',
                'emotion_distribution': {},
                'total_frames': 0,
                'total_faces_detected': 0,
                'processing_time': 0
            }
        
        # For demonstration, we'll generate random emotion distributions
        # In a real implementation, we would process each frame and aggregate results
        logger.info(f"Processing video: {video_path}")
        
        # Start timer
        start_time = time.time()
        
        try:
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Error opening video file: {video_path}")
                return {
                    'predominant_emotion': 'unknown',
                    'emotion_distribution': {},
                    'total_frames': 0,
                    'total_faces_detected': 0,
                    'processing_time': 0
                }
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # For demonstration, we'll process a subset of frames to simulate faster processing
            sample_rate = max(1, int(fps / 2))  # Process every half second
            emotions_detected = []
            total_faces = 0
            
            # Process frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_count % sample_rate == 0:
                    # Detect emotions
                    result = self.detect_emotion(frame)
                    
                    # Accumulate results
                    if result['primary_emotion'] != 'unknown':
                        emotions_detected.append(result['primary_emotion'])
                        total_faces += result['faces_detected']
                
                frame_count += 1
                
                # For demonstration, limit processing time
                if frame_count >= 100:
                    break
            
            # Release video capture
            cap.release()
            
            # Calculate predominant emotion
            emotion_counter = Counter(emotions_detected)
            predominant_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else 'unknown'
            
            # Calculate emotion distribution
            total_emotions = len(emotions_detected)
            emotion_distribution = {
                emotion: count / total_emotions for emotion, count in emotion_counter.items()
            } if total_emotions > 0 else {}
            
            # End timer
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Prepare result
            result = {
                'predominant_emotion': predominant_emotion,
                'emotion_distribution': emotion_distribution,
                'total_frames': total_frames,
                'processed_frames': frame_count,
                'sample_rate': sample_rate,
                'total_faces_detected': total_faces,
                'processing_time': processing_time,
                'duration': duration
            }
            
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            
            # End timer
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                'predominant_emotion': 'unknown',
                'emotion_distribution': {},
                'total_frames': 0,
                'total_faces_detected': 0,
                'processing_time': processing_time,
                'error': str(e)
            }

if __name__ == "__main__":
    # Test the emotion detector
    detector = EmotionDetector()
    
    # Create a test image with a solid color
    test_image = np.ones((300, 300, 3), dtype=np.uint8) * 200
    
    # Draw a simple face (circle)
    cv2.circle(test_image, (150, 150), 100, (100, 100, 100), -1)
    cv2.circle(test_image, (120, 120), 15, (0, 0, 0), -1)  # Left eye
    cv2.circle(test_image, (180, 120), 15, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(test_image, (150, 180), (50, 20), 0, 0, 180, (0, 0, 0), -1)  # Mouth
    
    # Detect emotions
    result = detector.detect_emotion(test_image)
    
    # Print results
    print("Primary Emotion:", result['primary_emotion'])
    print("Emotion Scores:", result['emotion_scores'])
    print("Faces Detected:", result['faces_detected'])
    
    # Draw emotions
    annotated_image = detector.draw_emotion(test_image, result)
    
    # Show image (if running interactively)
    # cv2.imshow("Emotion Detection Test", annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 