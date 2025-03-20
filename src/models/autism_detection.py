#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Autism Detection Model for children.
This module implements multiple machine learning models for detecting autism spectrum disorder in children
using behavioral, visual, and auditory features.
"""

import os
import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Set up logging
logger = logging.getLogger(__name__)

class AutismDetectionModel:
    """
    A class for detecting autism using multiple features.
    The model combines behavioral questionnaire data, gaze patterns, and vocal characteristics.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the autism detection model.
        
        Args:
            model_path (str, optional): Path to a pre-trained model. If None, a new model will be created.
        """
        self.model = None
        self.scaler = StandardScaler()
        
        # Important features based on research literature
        self.behavioral_features = [
            'eye_contact', 'responds_to_name', 'social_smile', 
            'interest_in_peers', 'pointing', 'showing', 'pretend_play',
            'follows_gaze', 'repetitive_behaviors', 'unusual_interests'
        ]
        
        self.visual_features = [
            'gaze_fixation_duration', 'gaze_transition_frequency',
            'face_looking_ratio', 'joint_attention_score'
        ]
        
        self.auditory_features = [
            'vocalization_frequency', 'prosody_variation',
            'speech_rate', 'articulation_quality'
        ]
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading autism detection model from {model_path}")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            logger.info("Creating a new autism detection model")
            self._create_model()
    
    def _create_model(self):
        """Create the machine learning model for autism detection."""
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced'
            ))
        ])
        
        logger.info("Autism detection model created successfully")
    
    def train(self, data, labels, test_size=0.2, optimize=False):
        """
        Train the autism detection model.
        
        Args:
            data (pandas.DataFrame): Feature data containing behavioral, visual, and auditory features
            labels (array-like): Target labels (1 for autism, 0 for typical development)
            test_size (float): Proportion of data to use for testing
            optimize (bool): Whether to perform hyperparameter optimization
            
        Returns:
            dict: Dictionary containing training and evaluation metrics
        """
        logger.info("Starting model training")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        if optimize:
            logger.info("Performing hyperparameter optimization")
            
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Model training completed with ROC AUC: {roc_auc:.4f}")
        
        return {
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc
        }
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path where the model will be saved
        """
        if self.model:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.error("No model to save")
    
    def predict(self, features):
        """
        Predict autism based on provided features.
        
        Args:
            features (dict or pandas.DataFrame): Dictionary or DataFrame containing feature values
            
        Returns:
            dict: Dictionary containing prediction results
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained")
        
        # Convert dictionary to DataFrame if necessary
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Ensure all required features are present
        required_features = self.behavioral_features + self.visual_features + self.auditory_features
        missing_features = set(required_features) - set(features.columns)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with zeros
            for feature in missing_features:
                features[feature] = 0
        
        # Reorder columns to match training data
        features = features[required_features]
        
        # Make prediction
        probability = self.model.predict_proba(features)[0, 1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Determine confidence level
        if probability <= 0.3 or probability >= 0.7:
            confidence = "high"
        elif probability <= 0.4 or probability >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Feature importance (for random forest)
        feature_importance = {}
        if hasattr(self.model['classifier'], 'feature_importances_'):
            importances = self.model['classifier'].feature_importances_
            feature_importance = dict(zip(required_features, importances))
        
        # Top contributing features
        if feature_importance:
            top_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        else:
            top_features = []
        
        result = {
            'prediction': prediction,
            'probability': float(probability),
            'confidence': confidence,
            'interpretation': "autism spectrum" if prediction == 1 else "typical development",
            'top_contributing_features': top_features
        }
        
        return result
    
    def generate_report(self, features):
        """
        Generate a detailed report based on the prediction.
        
        Args:
            features (dict or pandas.DataFrame): Dictionary or DataFrame containing feature values
            
        Returns:
            dict: Dictionary containing detailed report information
        """
        prediction_result = self.predict(features)
        
        # Convert dictionary to DataFrame if necessary
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Analyze behavioral patterns
        behavioral_scores = features[self.behavioral_features].iloc[0].to_dict()
        
        # Areas of concern (features with high values for autism prediction)
        if prediction_result['prediction'] == 1:
            areas_of_concern = [
                feature for feature, value in behavioral_scores.items() 
                if value >= 0.7
            ]
        else:
            areas_of_concern = []
        
        # Strengths (features with low values for autism prediction)
        if prediction_result['prediction'] == 1:
            strengths = [
                feature for feature, value in behavioral_scores.items() 
                if value <= 0.3
            ]
        else:
            strengths = list(behavioral_scores.keys())
        
        # Generate recommendations based on prediction
        if prediction_result['prediction'] == 1:
            if prediction_result['confidence'] == "high":
                recommendation = "Consult with a developmental pediatrician or child psychologist for a comprehensive assessment."
            else:
                recommendation = "Consider a follow-up screening in 3-6 months and monitor developmental milestones."
        else:
            recommendation = "Continue regular developmental monitoring as part of routine pediatric care."
        
        # Generate specific interventions based on areas of concern
        interventions = []
        
        if 'eye_contact' in areas_of_concern:
            interventions.append("Practice face-to-face interaction games")
        
        if 'responds_to_name' in areas_of_concern:
            interventions.append("Use child's name frequently in positive contexts")
        
        if 'social_smile' in areas_of_concern:
            interventions.append("Engage in playful facial expressions and mirroring activities")
        
        if 'repetitive_behaviors' in areas_of_concern:
            interventions.append("Redirect repetitive behaviors to functional activities with similar sensory feedback")
        
        # Combine all information into a comprehensive report
        report = {
            'prediction_summary': prediction_result,
            'behavioral_analysis': {
                'scores': behavioral_scores,
                'areas_of_concern': areas_of_concern,
                'strengths': strengths
            },
            'recommendations': {
                'primary_recommendation': recommendation,
                'suggested_interventions': interventions
            },
            'disclaimer': "This screening tool is not a diagnostic instrument. A formal diagnosis of autism spectrum disorder must be made by qualified healthcare professionals."
        }
        
        return report 