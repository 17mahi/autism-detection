#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for the Autism Detection and Emotion Analysis System.
This module provides functions to create visualizations for analysis results.
"""

import os
import logging
import numpy as np
import json
import base64
from io import BytesIO
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Visualizer:
    """
    A class to generate visualizations for autism and emotion analysis results.
    
    This is a placeholder implementation that returns base64-encoded dummy charts.
    In a production environment, this would generate actual visualizations.
    """
    
    def __init__(self):
        """Initialize the visualizer class."""
        logger.info("Initializing Visualizer")
    
    def create_emotion_pie_chart(self, emotion_data):
        """
        Create a pie chart of emotion distribution.
        
        Args:
            emotion_data (dict): Dictionary of emotions and their probabilities.
        
        Returns:
            str: Base64-encoded chart image or chart configuration JSON.
        """
        logger.info("Creating emotion pie chart")
        
        try:
            # In a real implementation, this would use matplotlib, plotly, or another 
            # visualization library to create a pie chart
            
            # For this placeholder, return a JSON configuration for a chart library
            chart_config = {
                "type": "pie",
                "data": {
                    "labels": list(emotion_data.keys()),
                    "datasets": [{
                        "data": [val * 100 for val in emotion_data.values()],
                        "backgroundColor": [
                            "#F1C40F", # happy
                            "#3498DB", # sad
                            "#E74C3C", # angry
                            "#2ECC71", # surprise
                            "#9B59B6", # fear
                            "#8E44AD", # disgust
                            "#95A5A6"  # neutral
                        ]
                    }]
                },
                "options": {
                    "title": {
                        "display": True,
                        "text": "Emotion Distribution"
                    }
                }
            }
            
            return json.dumps(chart_config)
            
        except Exception as e:
            logger.error(f"Error creating emotion pie chart: {str(e)}")
            return json.dumps({"error": "Failed to create chart"})
    
    def create_emotion_timeline(self, timeline_data):
        """
        Create a timeline visualization of emotion changes over time.
        
        Args:
            timeline_data (list): List of timestamped emotion data points.
        
        Returns:
            str: Base64-encoded chart image or chart configuration JSON.
        """
        logger.info("Creating emotion timeline chart")
        
        try:
            # Extract timestamps and emotion values
            timestamps = [point["timestamp"] for point in timeline_data]
            
            # Create datasets for each emotion
            datasets = []
            emotions = timeline_data[0]["emotions"].keys()
            
            colors = {
                "happy": "#F1C40F",
                "sad": "#3498DB",
                "angry": "#E74C3C",
                "surprise": "#2ECC71",
                "fear": "#9B59B6",
                "disgust": "#8E44AD",
                "neutral": "#95A5A6"
            }
            
            for emotion in emotions:
                data = [point["emotions"].get(emotion, 0) * 100 for point in timeline_data]
                datasets.append({
                    "label": emotion.capitalize(),
                    "data": data,
                    "borderColor": colors.get(emotion, "#000000"),
                    "backgroundColor": colors.get(emotion, "#000000") + "40",
                    "fill": False,
                    "tension": 0.4
                })
            
            # Create chart configuration
            chart_config = {
                "type": "line",
                "data": {
                    "labels": timestamps,
                    "datasets": datasets
                },
                "options": {
                    "scales": {
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Time (seconds)"
                            }
                        },
                        "y": {
                            "title": {
                                "display": True,
                                "text": "Emotion Intensity (%)"
                            },
                            "min": 0,
                            "max": 100
                        }
                    },
                    "title": {
                        "display": True,
                        "text": "Emotion Timeline"
                    }
                }
            }
            
            return json.dumps(chart_config)
            
        except Exception as e:
            logger.error(f"Error creating emotion timeline chart: {str(e)}")
            return json.dumps({"error": "Failed to create timeline"})
    
    def create_behavioral_radar_chart(self, behavioral_data):
        """
        Create a radar chart of behavioral assessment scores.
        
        Args:
            behavioral_data (dict): Dictionary containing behavioral assessment scores.
        
        Returns:
            str: Base64-encoded chart image or chart configuration JSON.
        """
        logger.info("Creating behavioral radar chart")
        
        try:
            # Extract scores
            scores = behavioral_data.get("scores", {})
            
            # Create chart configuration
            chart_config = {
                "type": "radar",
                "data": {
                    "labels": [key.replace("_", " ").title() for key in scores.keys()],
                    "datasets": [{
                        "label": "Score",
                        "data": list(scores.values()),
                        "backgroundColor": "rgba(54, 162, 235, 0.2)",
                        "borderColor": "rgb(54, 162, 235)",
                        "pointBackgroundColor": "rgb(54, 162, 235)",
                        "pointBorderColor": "#fff",
                        "pointHoverBackgroundColor": "#fff",
                        "pointHoverBorderColor": "rgb(54, 162, 235)"
                    }]
                },
                "options": {
                    "elements": {
                        "line": {
                            "tension": 0
                        }
                    },
                    "scales": {
                        "r": {
                            "angleLines": {
                                "display": True
                            },
                            "suggestedMin": 0,
                            "suggestedMax": 1
                        }
                    },
                    "title": {
                        "display": True,
                        "text": "Behavioral Assessment Profile"
                    }
                }
            }
            
            return json.dumps(chart_config)
            
        except Exception as e:
            logger.error(f"Error creating behavioral radar chart: {str(e)}")
            return json.dumps({"error": "Failed to create chart"})
    
    def create_summary_visualization(self, analysis_results):
        """
        Create a comprehensive visualization summarizing all analysis results.
        
        Args:
            analysis_results (dict): Complete analysis results.
        
        Returns:
            str: Base64-encoded chart image or chart configuration JSON.
        """
        logger.info("Creating summary visualization")
        
        try:
            # This would create a comprehensive dashboard visualization
            # For this placeholder, return a dashboard configuration
            
            dashboard_config = {
                "title": "Analysis Summary",
                "charts": [
                    {
                        "type": "gauge",
                        "title": "Autism Screening Score",
                        "value": analysis_results.get("prediction_summary", {}).get("probability", 0) * 100,
                        "min": 0,
                        "max": 100,
                        "thresholds": [30, 50, 70]
                    },
                    {
                        "type": "pie",
                        "title": "Emotion Distribution",
                        "data": analysis_results.get("emotions", {"neutral": 1.0})
                    },
                    {
                        "type": "radar",
                        "title": "Behavioral Profile",
                        "data": analysis_results.get("behavioral_analysis", {}).get("scores", {})
                    }
                ]
            }
            
            return json.dumps(dashboard_config)
            
        except Exception as e:
            logger.error(f"Error creating summary visualization: {str(e)}")
            return json.dumps({"error": "Failed to create summary visualization"})
    
    def __str__(self):
        return "Visualizer()"
        
    def __repr__(self):
        return self.__str__() 