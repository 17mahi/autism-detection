#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization module for the Autism Detection and Emotion Analysis System.
This module provides utilities for visualizing data and results.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Visualizer:
    """
    A class for visualizing data and results for autism detection and emotion analysis.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str, optional): Directory to save visualization outputs.
                If None, a default directory will be used.
        """
        if output_dir:
            self.output_dir = output_dir
        else:
            # Default to 'static/images' relative to this file
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static', 'images')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
        
        # Set default visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # Define color maps
        self.emotion_colors = {
            'happy': '#F1C40F',     # Yellow
            'sad': '#3498DB',       # Blue
            'angry': '#E74C3C',     # Red
            'surprise': '#2ECC71',  # Green
            'fear': '#9B59B6',      # Purple
            'disgust': '#8E44AD',   # Dark Purple
            'neutral': '#95A5A6'    # Gray
        }
        
        self.risk_level_colors = {
            'low': '#2ECC71',       # Green
            'medium': '#F1C40F',    # Yellow
            'high': '#E74C3C',      # Red
            'very_high': '#8E44AD'  # Purple
        }
    
    def plot_emotion_distribution(self, emotion_data, title='Emotion Distribution', filename=None):
        """
        Plot the distribution of emotions.
        
        Args:
            emotion_data (dict): Dictionary mapping emotions to their frequencies.
            title (str): Plot title.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create color list based on emotion names
            colors = [self.emotion_colors.get(emotion, '#333333') for emotion in emotion_data.keys()]
            
            # Plot bar chart
            bars = plt.bar(
                list(emotion_data.keys()), 
                list(emotion_data.values()),
                color=colors
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center', 
                    va='bottom',
                    fontweight='bold'
                )
            
            # Add labels and title
            plt.xlabel('Emotions')
            plt.ylabel('Frequency')
            plt.title(title)
            plt.ylim(0, 1.0)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=30)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Emotion distribution plot saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting emotion distribution: {str(e)}")
            plt.close()
            return None
    
    def plot_emotion_timeline(self, df, title='Emotion Timeline', filename=None):
        """
        Plot emotions over time.
        
        Args:
            df (pandas.DataFrame): DataFrame with columns for timestamps and emotion values.
            title (str): Plot title.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Get emotion columns (all columns except timestamp)
            emotion_columns = [col for col in df.columns if col != 'timestamp']
            
            # Plot each emotion as a line
            for emotion in emotion_columns:
                plt.plot(
                    df['timestamp'], 
                    df[emotion], 
                    label=emotion,
                    color=self.emotion_colors.get(emotion, None),
                    linewidth=2,
                    marker='o',
                    markersize=4
                )
            
            # Add labels and title
            plt.xlabel('Time')
            plt.ylabel('Intensity')
            plt.title(title)
            plt.legend()
            
            # Adjust x-axis for better readability
            plt.xticks(rotation=45)
            
            # Add grid
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Emotion timeline plot saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting emotion timeline: {str(e)}")
            plt.close()
            return None
    
    def plot_feature_importance(self, feature_importance, title='Feature Importance', filename=None):
        """
        Plot feature importance for autism detection.
        
        Args:
            feature_importance (dict): Dictionary mapping features to their importance scores.
            title (str): Plot title.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            feature_names = [item[0] for item in sorted_features]
            importance_scores = [item[1] for item in sorted_features]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot horizontal bar chart
            bars = plt.barh(
                feature_names, 
                importance_scores,
                color=sns.color_palette("viridis", len(feature_names))
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}',
                    ha='left', 
                    va='center',
                    fontweight='bold'
                )
            
            # Add labels and title
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.title(title)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            plt.close()
            return None
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title='Confusion Matrix', filename=None):
        """
        Plot a confusion matrix.
        
        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            labels (list, optional): List of label names.
            title (str): Plot title.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot confusion matrix as heatmap
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels if labels else 'auto',
                yticklabels=labels if labels else 'auto'
            )
            
            # Add labels and title
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(title)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix plot saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            plt.close()
            return None
    
    def plot_roc_curve(self, y_true, y_score, title='ROC Curve', filename=None):
        """
        Plot a ROC curve.
        
        Args:
            y_true (array-like): True binary labels.
            y_score (array-like): Predicted probabilities.
            title (str): Plot title.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Create figure
            plt.figure(figsize=(8, 8))
            
            # Plot ROC curve
            plt.plot(
                fpr, 
                tpr, 
                color='darkorange',
                lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})'
            )
            
            # Plot diagonal line (random classifier)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            # Add labels and title
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            
            # Set axis limits
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            
            # Add grid
            plt.grid(linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curve plot saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            plt.close()
            return None
    
    def plot_precision_recall_curve(self, y_true, y_score, title='Precision-Recall Curve', filename=None):
        """
        Plot a precision-recall curve.
        
        Args:
            y_true (array-like): True binary labels.
            y_score (array-like): Predicted probabilities.
            title (str): Plot title.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            
            # Create figure
            plt.figure(figsize=(8, 8))
            
            # Plot precision-recall curve
            plt.plot(
                recall, 
                precision, 
                color='green',
                lw=2, 
                label=f'PR curve (area = {pr_auc:.2f})'
            )
            
            # Add labels and title
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.legend(loc="lower left")
            
            # Set axis limits
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            
            # Add grid
            plt.grid(linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Precision-Recall curve plot saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting Precision-Recall curve: {str(e)}")
            plt.close()
            return None
    
    def plot_feature_distribution(self, data, feature, target=None, title=None, filename=None):
        """
        Plot the distribution of a feature, optionally grouped by a target variable.
        
        Args:
            data (pandas.DataFrame): DataFrame containing the data.
            feature (str): Feature to plot.
            target (str, optional): Target variable for grouping.
            title (str, optional): Plot title. If None, a default title is used.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Set default title if not provided
            if title is None:
                title = f'Distribution of {feature}'
                if target:
                    title += f' by {target}'
            
            # Plot distribution
            if target is None:
                # Single distribution
                sns.histplot(data=data, x=feature, kde=True)
            else:
                # Distribution grouped by target
                sns.histplot(data=data, x=feature, hue=target, kde=True, multiple="stack")
            
            # Add title
            plt.title(title)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature distribution plot saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting feature distribution: {str(e)}")
            plt.close()
            return None
    
    def plot_correlation_matrix(self, data, title='Correlation Matrix', filename=None):
        """
        Plot a correlation matrix of features.
        
        Args:
            data (pandas.DataFrame): DataFrame containing the features.
            title (str): Plot title.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Compute correlation matrix
            corr_matrix = data.corr()
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot correlation matrix as heatmap
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
                vmin=-1, 
                vmax=1,
                center=0
            )
            
            # Add title
            plt.title(title)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation matrix plot saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")
            plt.close()
            return None
    
    def visualize_emotion_on_image(self, image, results, output_path=None):
        """
        Draw emotion detection results on an image.
        
        Args:
            image (numpy.ndarray or str): Input image or path to image file.
            results (dict): Emotion detection results.
            output_path (str, optional): Path to save the visualized image.
                If None, the image is displayed.
            
        Returns:
            numpy.ndarray: Visualized image.
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    logger.error(f"Error loading image: {image}")
                    return None
            else:
                img = image.copy()
            
            # Check if results contain face and emotion information
            if 'faces_detected' not in results or results['faces_detected'] == 0:
                logger.info("No faces detected in the image")
                return img
            
            # Draw on image
            primary_emotion = results.get('primary_emotion', 'unknown')
            emotion_scores = results.get('emotion_scores', {})
            
            # If the results contain processed faces, use them
            if 'processed_faces' in results:
                for face in results['processed_faces']:
                    x, y, w, h = face['face_rect']
                    emotion = face['emotion']
                    score = face['score']
                    
                    # Get color for this emotion
                    color_hex = self.emotion_colors.get(emotion, '#FFFFFF')
                    # Convert hex to BGR (OpenCV format)
                    color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                    
                    # Draw rectangle around face
                    cv2.rectangle(img, (x, y), (x+w, y+h), color_bgr, 2)
                    
                    # Add emotion label
                    label = f"{emotion}: {score:.2f}"
                    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
            
            # If 'processed_faces' not available, use overall primary emotion
            else:
                # Get color for primary emotion
                color_hex = self.emotion_colors.get(primary_emotion, '#FFFFFF')
                # Convert hex to BGR (OpenCV format)
                color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                
                # Add emotion label at the top of the image
                label = f"Primary Emotion: {primary_emotion}"
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)
                
                # Add scores
                y_offset = 70
                for emotion, score in emotion_scores.items():
                    color_hex = self.emotion_colors.get(emotion, '#FFFFFF')
                    color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                    
                    label = f"{emotion}: {score:.2f}"
                    cv2.putText(img, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                    y_offset += 30
            
            # Save or display the image
            if output_path:
                cv2.imwrite(output_path, img)
                logger.info(f"Visualized image saved to {output_path}")
            
            return img
        
        except Exception as e:
            logger.error(f"Error visualizing emotion on image: {str(e)}")
            return None
    
    def create_emotion_pie_chart(self, emotion_data, title='Emotion Distribution', filename=None):
        """
        Create a pie chart of emotion distribution.
        
        Args:
            emotion_data (dict): Dictionary mapping emotions to their frequencies.
            title (str): Plot title.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Create color list based on emotion names
            colors = [self.emotion_colors.get(emotion, '#333333') for emotion in emotion_data.keys()]
            
            # Plot pie chart
            patches, texts, autotexts = plt.pie(
                list(emotion_data.values()), 
                labels=list(emotion_data.keys()),
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            
            # Style text
            for text in texts:
                text.set_fontsize(12)
            for autotext in autotexts:
                autotext.set_fontsize(12)
                autotext.set_fontweight('bold')
            
            # Add title
            plt.title(title, fontsize=16, pad=20)
            
            # Add equal aspect ratio to make the pie circular
            plt.axis('equal')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Emotion pie chart saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error creating emotion pie chart: {str(e)}")
            plt.close()
            return None
    
    def visualize_feature_space(self, X, y, labels=None, title='Feature Space Visualization', filename=None):
        """
        Visualize high-dimensional feature space using t-SNE.
        
        Args:
            X (array-like): Features matrix.
            y (array-like): Labels or targets.
            labels (list, optional): Label names for the legend.
            title (str): Plot title.
            filename (str, optional): Filename to save the plot. If None, the plot is displayed.
            
        Returns:
            str: Path to the saved plot file, or None if not saved.
        """
        try:
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot t-SNE result
            unique_labels = np.unique(y)
            for label in unique_labels:
                mask = (y == label)
                plt.scatter(
                    X_tsne[mask, 0], 
                    X_tsne[mask, 1], 
                    alpha=0.8, 
                    label=labels[label] if labels else label
                )
            
            # Add labels and title
            plt.xlabel('t-SNE Feature 1')
            plt.ylabel('t-SNE Feature 2')
            plt.title(title)
            plt.legend()
            
            # Add grid
            plt.grid(linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display the plot
            if filename:
                file_path = os.path.join(self.output_dir, filename)
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature space visualization saved to {file_path}")
                plt.close()
                return file_path
            else:
                plt.show()
                return None
        
        except Exception as e:
            logger.error(f"Error visualizing feature space: {str(e)}")
            plt.close()
            return None

if __name__ == "__main__":
    # Test the visualizer
    visualizer = Visualizer()
    
    # Test emotion distribution
    emotion_data = {
        'happy': 0.45,
        'sad': 0.15,
        'angry': 0.05,
        'surprise': 0.10,
        'fear': 0.05,
        'disgust': 0.05,
        'neutral': 0.15
    }
    
    # Test pie chart
    visualizer.create_emotion_pie_chart(emotion_data, title='Test Emotion Distribution', filename='test_emotion_pie.png')
    
    # Test feature importance
    feature_importance = {
        'social_interaction': 0.35,
        'communication': 0.25,
        'repetitive_behavior': 0.20,
        'sensory_sensitivity': 0.12,
        'eye_contact': 0.05,
        'attention_to_detail': 0.03
    }
    
    visualizer.plot_feature_importance(feature_importance, title='Test Feature Importance', filename='test_feature_importance.png')
    
    print("Test visualization complete. Check the output directory for generated plots.") 