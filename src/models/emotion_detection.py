#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Emotion Detection Model for children.
This module implements a deep learning model for detecting emotions from facial expressions.
"""

import os
import logging
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up logging
logger = logging.getLogger(__name__)

class EmotionDetectionModel:
    """
    A class for detecting emotions from facial expressions using deep learning.
    The model is trained to detect 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion detection model.
        
        Args:
            model_path (str, optional): Path to a pre-trained model. If None, a new model will be created.
        """
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading emotion detection model from {model_path}")
            self.model = load_model(model_path)
        else:
            logger.info("Creating a new emotion detection model")
            self._create_model()
    
    def _create_model(self):
        """Create the CNN model architecture for emotion detection."""
        self.model = Sequential()
        
        # First convolutional layer
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        
        # Second convolutional layer
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        
        # Flattening and fully connected layers
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))  # 7 emotions
        
        # Compile the model
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        logger.info("Emotion detection model created successfully")
    
    def train(self, train_dir, validation_dir, epochs=50, batch_size=64):
        """
        Train the emotion detection model.
        
        Args:
            train_dir (str): Directory containing training data
            validation_dir (str): Directory containing validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History object containing training metrics
        """
        logger.info("Starting model training")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical'
        )
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        logger.info("Model training completed")
        return history
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path where the model will be saved
        """
        if self.model:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.error("No model to save")
    
    def detect_emotion(self, image):
        """
        Detect emotions in an image.
        
        Args:
            image (numpy.ndarray): Image in which to detect emotions
            
        Returns:
            list: List of dictionaries containing face locations and emotions
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face ROI
            roi_gray = gray[y:y+h, x:x+w]
            
            try:
                # Resize to 48x48 (model input size)
                roi_gray = cv2.resize(roi_gray, (48, 48))
                
                # Normalize and reshape
                roi_gray = roi_gray / 255.0
                roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
                
                # Predict emotion
                prediction = self.model.predict(roi_gray)[0]
                
                # Get the label with highest probability
                emotion_idx = np.argmax(prediction)
                emotion = self.emotions[emotion_idx]
                
                # Store result
                results.append({
                    'position': (x, y, w, h),
                    'emotion': emotion,
                    'confidence': float(prediction[emotion_idx]),
                    'all_emotions': {self.emotions[i]: float(prediction[i]) for i in range(len(self.emotions))}
                })
                
            except Exception as e:
                logger.error(f"Error processing face: {str(e)}")
        
        return results
    
    def analyze_video(self, video_path, output_path=None, frame_interval=30):
        """
        Analyze emotions in a video file.
        
        Args:
            video_path (str): Path to the video file
            output_path (str, optional): Path to save the output video
            frame_interval (int): Process every nth frame
            
        Returns:
            dict: Summary of emotions detected throughout the video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize counters
        frame_idx = 0
        emotion_counts = {emotion: 0 for emotion in self.emotions}
        total_faces = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every nth frame to improve performance
            if frame_idx % frame_interval == 0:
                # Detect emotions in the frame
                results = self.detect_emotion(frame)
                
                # Update counters
                for result in results:
                    emotion_counts[result['emotion']] += 1
                    total_faces += 1
                
                # Draw results on frame
                for result in results:
                    x, y, w, h = result['position']
                    emotion = result['emotion']
                    confidence = result['confidence']
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display emotion and confidence
                    text = f"{emotion} ({confidence:.2f})"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Write frame to output video
            if output_path:
                out.write(frame)
            
            frame_idx += 1
            
            # Log progress
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames")
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        
        # Calculate emotion distribution
        emotion_distribution = {emotion: count/total_faces if total_faces > 0 else 0 
                               for emotion, count in emotion_counts.items()}
        
        # Determine predominant emotion
        predominant_emotion = max(emotion_distribution.items(), key=lambda x: x[1])[0] if total_faces > 0 else None
        
        return {
            'total_frames_processed': frame_idx // frame_interval,
            'total_faces_detected': total_faces,
            'emotion_counts': emotion_counts,
            'emotion_distribution': emotion_distribution,
            'predominant_emotion': predominant_emotion
        } 