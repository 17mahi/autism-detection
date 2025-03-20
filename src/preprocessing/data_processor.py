#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing module for autism detection and emotion analysis.
This module handles preprocessing of various data types including
questionnaire data, video data, and audio data.
"""

import os
import logging
import numpy as np
import pandas as pd
import cv2
import json
from tqdm import tqdm
import mediapipe as mp

# Set up logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class for processing various data types for autism detection and emotion analysis.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        # Initialize MediaPipe for face mesh and hand tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Set up paths for storing processed data
        self.processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'data', 'processed')
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)
    
    def process_questionnaire_data(self, data_path, output_path=None):
        """
        Process questionnaire data for autism detection.
        
        Args:
            data_path (str): Path to the questionnaire data file (CSV)
            output_path (str, optional): Path to save the processed data
            
        Returns:
            pandas.DataFrame: Processed questionnaire data
        """
        logger.info(f"Processing questionnaire data from {data_path}")
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            
            # Perform basic cleaning
            # Remove rows with excessive missing values
            data = data.dropna(thresh=len(data.columns) * 0.7)
            
            # Handle remaining missing values
            for column in data.columns:
                if data[column].dtype in [np.float64, np.int64]:
                    # Fill numeric columns with mean
                    data[column] = data[column].fillna(data[column].mean())
                else:
                    # Fill categorical columns with mode
                    data[column] = data[column].fillna(data[column].mode()[0])
            
            # Normalize numerical features to 0-1 range
            numerical_cols = data.select_dtypes(include=[np.float64, np.int64]).columns
            for col in numerical_cols:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    data[col] = (data[col] - min_val) / (max_val - min_val)
            
            # Convert categorical features to one-hot encoding
            categorical_cols = data.select_dtypes(include=['object']).columns
            if not categorical_cols.empty:
                data = pd.get_dummies(data, columns=categorical_cols)
            
            # Save processed data if output path is provided
            if output_path:
                data.to_csv(output_path, index=False)
                logger.info(f"Processed data saved to {output_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing questionnaire data: {str(e)}")
            raise
    
    def extract_facial_landmarks(self, video_path, output_path=None, sample_rate=30):
        """
        Extract facial landmarks from video data for emotion and gaze analysis.
        
        Args:
            video_path (str): Path to the video file
            output_path (str, optional): Path to save the extracted landmarks
            sample_rate (int): Process every nth frame
            
        Returns:
            dict: Dictionary containing extracted facial landmarks
        """
        logger.info(f"Extracting facial landmarks from {video_path}")
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Unable to open video file: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            landmarks_data = {
                'video_path': video_path,
                'fps': float(fps),
                'total_frames': frame_count,
                'frames': []
            }
            
            # Initialize face mesh
            with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:
                
                frame_idx = 0
                success_count = 0
                
                # Process video frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Process every nth frame
                    if frame_idx % sample_rate == 0:
                        # Convert to RGB for MediaPipe
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process with MediaPipe
                        results = face_mesh.process(frame_rgb)
                        
                        frame_data = {
                            'frame_idx': frame_idx,
                            'timestamp': frame_idx / fps,
                            'landmarks': None
                        }
                        
                        if results.multi_face_landmarks:
                            # Extract landmark coordinates
                            face_landmarks = results.multi_face_landmarks[0]
                            landmarks = []
                            
                            for landmark in face_landmarks.landmark:
                                landmarks.append({
                                    'x': landmark.x,
                                    'y': landmark.y,
                                    'z': landmark.z
                                })
                            
                            frame_data['landmarks'] = landmarks
                            success_count += 1
                        
                        landmarks_data['frames'].append(frame_data)
                    
                    frame_idx += 1
                    
                    # Log progress periodically
                    if frame_idx % 100 == 0:
                        logger.info(f"Processed {frame_idx}/{frame_count} frames")
            
            # Release video capture
            cap.release()
            
            # Add summary statistics
            landmarks_data['success_rate'] = success_count / len(landmarks_data['frames']) if landmarks_data['frames'] else 0
            
            # Save extracted landmarks if output path is provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(landmarks_data, f)
                logger.info(f"Landmarks data saved to {output_path}")
            
            return landmarks_data
            
        except Exception as e:
            logger.error(f"Error extracting facial landmarks: {str(e)}")
            raise
    
    def extract_behavioral_features(self, landmarks_data):
        """
        Extract behavioral features from facial landmarks data.
        
        Args:
            landmarks_data (dict): Dictionary containing facial landmarks data
            
        Returns:
            dict: Dictionary containing extracted behavioral features
        """
        logger.info("Extracting behavioral features from landmarks data")
        
        try:
            # Eye contact features
            eye_contact_scores = []
            gaze_transitions = []
            face_looking_ratios = []
            
            frames_with_landmarks = [frame for frame in landmarks_data['frames'] if frame['landmarks']]
            
            if not frames_with_landmarks:
                logger.warning("No frames with landmarks found")
                return {
                    'eye_contact': 0,
                    'gaze_fixation_duration': 0,
                    'gaze_transition_frequency': 0,
                    'face_looking_ratio': 0,
                    'joint_attention_score': 0
                }
            
            # Process each frame with landmarks
            for i, frame in enumerate(frames_with_landmarks):
                landmarks = frame['landmarks']
                
                # Eye landmarks (indexes based on MediaPipe Face Mesh)
                left_eye = [landmarks[33], landmarks[133], landmarks[160], landmarks[159], landmarks[158], landmarks[144]]
                right_eye = [landmarks[362], landmarks[263], landmarks[387], landmarks[386], landmarks[385], landmarks[373]]
                
                # Calculate eye center
                left_eye_center = np.mean([[p['x'], p['y'], p['z']] for p in left_eye], axis=0)
                right_eye_center = np.mean([[p['x'], p['y'], p['z']] for p in right_eye], axis=0)
                
                # Calculate gaze direction (simplified)
                # In a real implementation, this would be more sophisticated
                gaze_direction = np.array([
                    (left_eye_center[0] + right_eye_center[0]) / 2,
                    (left_eye_center[1] + right_eye_center[1]) / 2,
                    (left_eye_center[2] + right_eye_center[2]) / 2
                ])
                
                # Normalize gaze direction
                gaze_magnitude = np.linalg.norm(gaze_direction)
                if gaze_magnitude > 0:
                    gaze_direction = gaze_direction / gaze_magnitude
                
                # Estimate eye contact (z-component indicates forward gaze)
                eye_contact_score = abs(gaze_direction[2])
                eye_contact_scores.append(eye_contact_score)
                
                # Calculate gaze transitions (changes in gaze direction)
                if i > 0:
                    prev_landmarks = frames_with_landmarks[i-1]['landmarks']
                    prev_left_eye = [prev_landmarks[33], prev_landmarks[133], prev_landmarks[160], 
                                    prev_landmarks[159], prev_landmarks[158], prev_landmarks[144]]
                    prev_right_eye = [prev_landmarks[362], prev_landmarks[263], prev_landmarks[387], 
                                     prev_landmarks[386], prev_landmarks[385], prev_landmarks[373]]
                    
                    prev_left_eye_center = np.mean([[p['x'], p['y'], p['z']] for p in prev_left_eye], axis=0)
                    prev_right_eye_center = np.mean([[p['x'], p['y'], p['z']] for p in prev_right_eye], axis=0)
                    
                    prev_gaze_direction = np.array([
                        (prev_left_eye_center[0] + prev_right_eye_center[0]) / 2,
                        (prev_left_eye_center[1] + prev_right_eye_center[1]) / 2,
                        (prev_left_eye_center[2] + prev_right_eye_center[2]) / 2
                    ])
                    
                    prev_gaze_magnitude = np.linalg.norm(prev_gaze_direction)
                    if prev_gaze_magnitude > 0:
                        prev_gaze_direction = prev_gaze_direction / prev_gaze_magnitude
                    
                    # Calculate change in gaze direction
                    gaze_change = np.linalg.norm(gaze_direction - prev_gaze_direction)
                    gaze_transitions.append(gaze_change)
                
                # Calculate face looking ratio (how much time spent looking at faces)
                # This is a simplified version; in reality, it would check if gaze intersects with detected faces
                face_looking_ratio = eye_contact_score
                face_looking_ratios.append(face_looking_ratio)
            
            # Calculate average eye contact score
            avg_eye_contact = np.mean(eye_contact_scores) if eye_contact_scores else 0
            
            # Calculate gaze fixation duration (inverse of transition frequency)
            gaze_transition_freq = np.mean(gaze_transitions) if gaze_transitions else 0
            gaze_fixation_duration = 1 - min(gaze_transition_freq, 1)
            
            # Calculate average face looking ratio
            avg_face_looking_ratio = np.mean(face_looking_ratios) if face_looking_ratios else 0
            
            # Calculate joint attention score (simplified)
            # In reality, this would involve more complex gaze following analysis
            joint_attention_score = avg_eye_contact * 0.7 + avg_face_looking_ratio * 0.3
            
            # Compile all features
            behavioral_features = {
                'eye_contact': float(avg_eye_contact),
                'gaze_fixation_duration': float(gaze_fixation_duration),
                'gaze_transition_frequency': float(gaze_transition_freq),
                'face_looking_ratio': float(avg_face_looking_ratio),
                'joint_attention_score': float(joint_attention_score)
            }
            
            return behavioral_features
            
        except Exception as e:
            logger.error(f"Error extracting behavioral features: {str(e)}")
            raise
    
    def extract_autism_features_from_video(self, video_path, output_path=None):
        """
        Extract features relevant to autism detection from video.
        
        Args:
            video_path (str): Path to the video file
            output_path (str, optional): Path to save the extracted features
            
        Returns:
            dict: Dictionary containing extracted features
        """
        logger.info(f"Extracting autism features from video: {video_path}")
        
        try:
            # Extract facial landmarks
            landmarks_file = os.path.join(self.processed_data_dir, f"{os.path.basename(video_path)}_landmarks.json")
            landmarks_data = self.extract_facial_landmarks(video_path, landmarks_file)
            
            # Extract behavioral features from landmarks
            behavioral_features = self.extract_behavioral_features(landmarks_data)
            
            # For now, we'll use placeholder values for other features
            # In a complete implementation, these would be extracted from the video
            autism_features = {
                # Behavioral features from landmarks
                'eye_contact': behavioral_features['eye_contact'],
                'gaze_fixation_duration': behavioral_features['gaze_fixation_duration'],
                'gaze_transition_frequency': behavioral_features['gaze_transition_frequency'],
                'face_looking_ratio': behavioral_features['face_looking_ratio'],
                'joint_attention_score': behavioral_features['joint_attention_score'],
                
                # Placeholder values for other features
                # These would be more sophisticated in a real implementation
                'responds_to_name': 0.5,
                'social_smile': 0.5,
                'interest_in_peers': 0.5,
                'pointing': 0.5,
                'showing': 0.5,
                'pretend_play': 0.5,
                'follows_gaze': 0.5,
                'repetitive_behaviors': 0.5,
                'unusual_interests': 0.5,
                'vocalization_frequency': 0.5,
                'prosody_variation': 0.5,
                'speech_rate': 0.5,
                'articulation_quality': 0.5
            }
            
            # Save extracted features if output path is provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(autism_features, f)
                logger.info(f"Autism features saved to {output_path}")
            
            return autism_features
            
        except Exception as e:
            logger.error(f"Error extracting autism features: {str(e)}")
            raise