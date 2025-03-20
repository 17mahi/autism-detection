#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Autism Detector module for analyzing questionnaire responses and video data.
This is a placeholder implementation that would be replaced with actual ML models.
"""

import os
import logging
import json
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutismDetector:
    """
    A class for analyzing questionnaire responses to provide autism screening results.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the autism detector with pre-trained models.
        
        Args:
            model_path (str, optional): Path to the autism detection model.
                If None, default models will be used.
        """
        # Define feature categories and their weights
        self.feature_categories = {
            'social_interaction': 0.25,
            'communication': 0.20,
            'repetitive_behavior': 0.20,
            'sensory_sensitivity': 0.15,
            'eye_contact': 0.10,
            'attention_to_detail': 0.10
        }
        
        # Load models
        try:
            # In a real implementation, we would load a machine learning model
            # For demonstration purposes, we'll use a rule-based approach
            
            # Load question mapping
            self.question_mapping_path = os.path.join(
                os.path.dirname(__file__), 
                'question_mapping.json'
            )
            
            if os.path.exists(self.question_mapping_path):
                with open(self.question_mapping_path, 'r') as f:
                    self.question_mapping = json.load(f)
                logger.info("Question mapping loaded successfully")
            else:
                # Default question mapping
                self.question_mapping = self.get_default_question_mapping()
                logger.warning("Using default question mapping")
            
            # Risk thresholds
            self.risk_thresholds = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
            }
            
            logger.warning("Using mock autism detector (rule-based)")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Error initializing autism detector: {str(e)}")
            self.model_loaded = False
    
    def get_default_question_mapping(self):
        """
        Get the default question mapping for autism screening.
        
        Returns:
            dict: Default question mapping.
        """
        return {
            # Sample question mappings based on AQ and M-CHAT
            # Format: "question_id": {"category": "category_name", "weight": weight, "positive_response": response}
            
            # Social interaction questions
            "q1": {"category": "social_interaction", "weight": 1.0, "positive_response": "rarely"},
            "q2": {"category": "social_interaction", "weight": 0.8, "positive_response": "rarely"},
            "q3": {"category": "social_interaction", "weight": 0.9, "positive_response": "rarely"},
            
            # Communication questions
            "q4": {"category": "communication", "weight": 1.0, "positive_response": "rarely"},
            "q5": {"category": "communication", "weight": 0.8, "positive_response": "rarely"},
            "q6": {"category": "communication", "weight": 0.9, "positive_response": "rarely"},
            
            # Repetitive behavior questions
            "q7": {"category": "repetitive_behavior", "weight": 1.0, "positive_response": "often"},
            "q8": {"category": "repetitive_behavior", "weight": 0.8, "positive_response": "often"},
            "q9": {"category": "repetitive_behavior", "weight": 0.9, "positive_response": "often"},
            
            # Sensory sensitivity questions
            "q10": {"category": "sensory_sensitivity", "weight": 1.0, "positive_response": "often"},
            "q11": {"category": "sensory_sensitivity", "weight": 0.8, "positive_response": "often"},
            "q12": {"category": "sensory_sensitivity", "weight": 0.9, "positive_response": "often"},
            
            # Eye contact questions
            "q13": {"category": "eye_contact", "weight": 1.0, "positive_response": "rarely"},
            "q14": {"category": "eye_contact", "weight": 0.8, "positive_response": "rarely"},
            
            # Attention to detail questions
            "q15": {"category": "attention_to_detail", "weight": 1.0, "positive_response": "often"},
            "q16": {"category": "attention_to_detail", "weight": 0.8, "positive_response": "often"}
        }
    
    def analyze_screening(self, data):
        """
        Analyze autism screening questionnaire responses.
        
        Args:
            data (dict): Questionnaire responses.
                Format: {"q1": "response", "q2": "response", ...}
                
        Returns:
            dict: Dictionary containing screening results.
        """
        if not self.model_loaded:
            logger.error("Autism detector model not loaded")
            return {
                'risk_score': 0.0,
                'risk_level': 'unknown',
                'feature_importance': {},
                'recommendations': []
            }
        
        try:
            # Validate data
            if not data or not isinstance(data, dict):
                logger.error("Invalid questionnaire data format")
                return {
                    'risk_score': 0.0,
                    'risk_level': 'unknown',
                    'feature_importance': {},
                    'recommendations': [],
                    'error': 'Invalid data format'
                }
            
            # Calculate category scores
            category_scores = defaultdict(float)
            category_weights = defaultdict(float)
            
            for question_id, response in data.items():
                if question_id in self.question_mapping:
                    question_info = self.question_mapping[question_id]
                    category = question_info["category"]
                    weight = question_info["weight"]
                    positive_response = question_info["positive_response"]
                    
                    # Calculate score (1.0 if response matches positive_response, 0.0 otherwise)
                    # More sophisticated scoring could be implemented here
                    if response == positive_response:
                        score = 1.0
                    else:
                        # For simplicity, we'll use a linear scale for other responses
                        response_values = {"never": 0.0, "rarely": 0.33, "sometimes": 0.5, "often": 0.67, "always": 1.0}
                        positive_value = response_values[positive_response]
                        response_value = response_values.get(response, 0.5)
                        
                        # Calculate score based on distance from positive response
                        if positive_value > 0.5:  # "often" or "always" is positive
                            score = response_value
                        else:  # "never" or "rarely" is positive
                            score = 1.0 - response_value
                    
                    # Add weighted score to category
                    category_scores[category] += score * weight
                    category_weights[category] += weight
            
            # Normalize category scores
            for category in category_scores:
                if category_weights[category] > 0:
                    category_scores[category] /= category_weights[category]
            
            # Calculate overall risk score
            risk_score = 0.0
            feature_importance = {}
            
            for category, weight in self.feature_categories.items():
                if category in category_scores:
                    risk_score += category_scores[category] * weight
                    feature_importance[category] = category_scores[category]
            
            # Determine risk level
            if risk_score < self.risk_thresholds['low']:
                risk_level = 'low'
            elif risk_score < self.risk_thresholds['medium']:
                risk_level = 'medium'
            elif risk_score < self.risk_thresholds['high']:
                risk_level = 'high'
            else:
                risk_level = 'very_high'
            
            # Generate recommendations
            recommendations = self.generate_recommendations(risk_level, feature_importance)
            
            # Prepare result
            result = {
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'feature_importance': feature_importance,
                'recommendations': recommendations
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing screening data: {str(e)}")
            return {
                'risk_score': 0.0,
                'risk_level': 'unknown',
                'feature_importance': {},
                'recommendations': [],
                'error': str(e)
            }
    
    def generate_recommendations(self, risk_level, feature_importance):
        """
        Generate recommendations based on risk level and feature importance.
        
        Args:
            risk_level (str): Risk level ('low', 'medium', 'high', 'very_high').
            feature_importance (dict): Dictionary of feature importance scores.
            
        Returns:
            list: List of recommendation strings.
        """
        recommendations = []
        
        # General recommendations based on risk level
        if risk_level == 'low':
            recommendations.append("Continue monitoring child's development and behavior.")
            recommendations.append("Engage in interactive play and social activities.")
            recommendations.append("No immediate clinical assessment needed, but follow up with regular developmental screenings.")
        
        elif risk_level == 'medium':
            recommendations.append("Consider scheduling a follow-up with a pediatrician for additional screening.")
            recommendations.append("Monitor social development more closely over the next few months.")
            recommendations.append("Increase engagement in interactive social play activities.")
            recommendations.append("Consider screening for other developmental or communication issues.")
        
        elif risk_level == 'high' or risk_level == 'very_high':
            recommendations.append("Consult with a healthcare professional specializing in developmental disorders.")
            recommendations.append("Request a comprehensive evaluation by a multidisciplinary team.")
            recommendations.append("Explore early intervention services in your area.")
            recommendations.append("Connect with parent support groups for autism spectrum disorders.")
        
        # Specific recommendations based on feature importance
        high_importance_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, score in high_importance_features[:3]:
            if score > 0.5:
                if feature == 'social_interaction':
                    recommendations.append("Focus on activities that encourage social interaction with peers.")
                    recommendations.append("Practice taking turns and sharing during play.")
                
                elif feature == 'communication':
                    recommendations.append("Encourage communication through visual supports and clear language.")
                    recommendations.append("Consider assessment by a speech-language pathologist.")
                
                elif feature == 'repetitive_behavior':
                    recommendations.append("Create a structured routine with visual schedules to reduce anxiety.")
                    recommendations.append("Gradually introduce small changes to routines to build flexibility.")
                
                elif feature == 'sensory_sensitivity':
                    recommendations.append("Create a sensory-friendly environment that minimizes overwhelming stimuli.")
                    recommendations.append("Consider occupational therapy for sensory integration support.")
                
                elif feature == 'eye_contact':
                    recommendations.append("Practice face-to-face interaction during enjoyable activities.")
                    recommendations.append("Use animated facial expressions during communication.")
                
                elif feature == 'attention_to_detail':
                    recommendations.append("Leverage attention to detail in learning activities.")
                    recommendations.append("Provide opportunities for focused interest development.")
        
        return recommendations

if __name__ == "__main__":
    # Test the autism detector
    detector = AutismDetector()
    
    # Sample questionnaire responses
    sample_responses = {
        "q1": "sometimes",
        "q2": "rarely",
        "q3": "rarely",
        "q4": "sometimes",
        "q5": "sometimes",
        "q6": "often",
        "q7": "often",
        "q8": "sometimes",
        "q9": "often",
        "q10": "sometimes",
        "q11": "often",
        "q12": "rarely",
        "q13": "rarely",
        "q14": "sometimes",
        "q15": "often",
        "q16": "sometimes"
    }
    
    # Analyze screening
    result = detector.analyze_screening(sample_responses)
    
    # Print results
    print("Risk Score:", result['risk_score'])
    print("Risk Level:", result['risk_level'])
    print("Feature Importance:")
    for feature, importance in result['feature_importance'].items():
        print(f"  {feature}: {importance:.2f}")
    print("\nRecommendations:")
    for recommendation in result['recommendations']:
        print(f"  - {recommendation}") 