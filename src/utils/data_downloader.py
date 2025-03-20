#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data downloader utility for autism detection and emotion analysis.
This module provides functions to download and prepare datasets for model training.
"""

import os
import logging
import requests
import zipfile
import tarfile
import shutil
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class DataDownloader:
    """
    A class for downloading and preparing datasets for the project.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the data downloader.
        
        Args:
            data_dir (str, optional): Directory to store downloaded data
        """
        if data_dir is None:
            # Use default data directory
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'data')
        else:
            self.data_dir = data_dir
            
        # Create data directories if they don't exist
        os.makedirs(os.path.join(self.data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'processed'), exist_ok=True)
        
        # Define URLs for datasets
        self.datasets = {
            'fer2013': {
                'url': 'https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data',
                'description': 'Facial Expression Recognition dataset with 35,887 grayscale images of faces',
                'type': 'emotion',
                'kaggle': True
            },
            'autism_screening': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00419/Autism-Adult-Data.arff',
                'description': 'Autism Screening Questionnaire dataset from UCI ML Repository',
                'type': 'autism',
                'kaggle': False
            },
            'child_autism': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00419/Autism-Child-Data.arff',
                'description': 'Child Autism Screening dataset from UCI ML Repository',
                'type': 'autism',
                'kaggle': False
            },
            'ravdess': {
                'url': 'https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip',
                'description': 'Ryerson Audio-Visual Database of Emotional Speech and Song',
                'type': 'emotion_audio',
                'kaggle': False
            }
        }
    
    def download_file(self, url, output_path, kaggle=False):
        """
        Download a file from a URL.
        
        Args:
            url (str): URL to download from
            output_path (str): Path to save the downloaded file
            kaggle (bool): Whether this is a Kaggle dataset requiring authentication
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if kaggle:
                logger.info("Kaggle datasets require authentication and the Kaggle API.")
                logger.info("Please install the Kaggle API: pip install kaggle")
                logger.info("Then download the dataset manually from:")
                logger.info(url)
                return False
            
            logger.info(f"Downloading from {url} to {output_path}")
            
            # Stream the download with progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(output_path, 'wb') as file, tqdm(
                desc=os.path.basename(output_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    file.write(data)
                    bar.update(len(data))
            
            logger.info(f"Download completed: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            return False
    
    def extract_archive(self, archive_path, extract_path):
        """
        Extract an archive file.
        
        Args:
            archive_path (str): Path to the archive file
            extract_path (str): Path to extract to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Extracting {archive_path} to {extract_path}")
            
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_path)
            elif archive_path.endswith('.tar'):
                with tarfile.open(archive_path, 'r:') as tar_ref:
                    tar_ref.extractall(extract_path)
            else:
                logger.error(f"Unsupported archive format: {archive_path}")
                return False
            
            logger.info(f"Extraction completed: {extract_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error extracting archive: {str(e)}")
            return False
    
    def download_dataset(self, dataset_name):
        """
        Download a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset to download
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset = self.datasets[dataset_name]
        output_dir = os.path.join(self.data_dir, 'raw', dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{dataset_name}.{'zip' if '.zip' in dataset['url'] else 'arff'}")
        
        success = self.download_file(dataset['url'], output_file, dataset.get('kaggle', False))
        
        if success and (output_file.endswith('.zip') or output_file.endswith('.tar.gz') or output_file.endswith('.tgz')):
            success = self.extract_archive(output_file, output_dir)
        
        return success
    
    def download_all_datasets(self):
        """
        Download all available datasets.
        
        Returns:
            dict: Dictionary of dataset names and their download status
        """
        results = {}
        
        for dataset_name in self.datasets:
            logger.info(f"Downloading dataset: {dataset_name}")
            results[dataset_name] = self.download_dataset(dataset_name)
        
        return results
    
    def convert_arff_to_csv(self, arff_path, csv_path):
        """
        Convert an ARFF file to CSV format.
        
        Args:
            arff_path (str): Path to the ARFF file
            csv_path (str): Path to save the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import pandas as pd
            from scipy.io import arff
            
            logger.info(f"Converting {arff_path} to {csv_path}")
            
            # Load ARFF file
            data, meta = arff.loadarff(arff_path)
            
            # Convert to DataFrame and save as CSV
            df = pd.DataFrame(data)
            
            # Convert bytes columns to strings (common in ARFF files)
            for col in df.columns:
                if df[col].dtype == object:  # Object dtype often indicates bytes
                    df[col] = df[col].str.decode('utf-8')
            
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Conversion completed: {csv_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error converting ARFF to CSV: {str(e)}")
            return False
    
    def prepare_autism_datasets(self):
        """
        Prepare autism datasets for use in the project.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Process adult autism dataset
            adult_arff = os.path.join(self.data_dir, 'raw', 'autism_screening', 'Autism-Adult-Data.arff')
            adult_csv = os.path.join(self.data_dir, 'processed', 'autism_adult.csv')
            
            if os.path.exists(adult_arff):
                self.convert_arff_to_csv(adult_arff, adult_csv)
            
            # Process child autism dataset
            child_arff = os.path.join(self.data_dir, 'raw', 'child_autism', 'Autism-Child-Data.arff')
            child_csv = os.path.join(self.data_dir, 'processed', 'autism_child.csv')
            
            if os.path.exists(child_arff):
                self.convert_arff_to_csv(child_arff, child_csv)
            
            return True
        
        except Exception as e:
            logger.error(f"Error preparing autism datasets: {str(e)}")
            return False
    
    def prepare_emotion_datasets(self):
        """
        Prepare emotion datasets for use in the project.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # FER2013 data is in a specific format from Kaggle
            fer_dir = os.path.join(self.data_dir, 'raw', 'fer2013')
            
            if not os.path.exists(fer_dir):
                logger.warning("FER2013 dataset not found. Please download manually from Kaggle.")
                return False
            
            # RAVDESS audio dataset
            ravdess_dir = os.path.join(self.data_dir, 'raw', 'ravdess')
            ravdess_processed = os.path.join(self.data_dir, 'processed', 'ravdess')
            
            if os.path.exists(ravdess_dir):
                os.makedirs(ravdess_processed, exist_ok=True)
                
                # Organize audio files by emotion
                emotion_map = {
                    '01': 'neutral',
                    '02': 'calm',
                    '03': 'happy',
                    '04': 'sad',
                    '05': 'angry',
                    '06': 'fearful',
                    '07': 'disgust',
                    '08': 'surprised'
                }
                
                # Create directories for each emotion
                for emotion in emotion_map.values():
                    os.makedirs(os.path.join(ravdess_processed, emotion), exist_ok=True)
                
                # Find all .wav files
                for root, _, files in os.walk(ravdess_dir):
                    for file in files:
                        if file.endswith('.wav'):
                            # RAVDESS filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
                            parts = file.split('-')
                            if len(parts) >= 7:
                                emotion_code = parts[2]
                                if emotion_code in emotion_map:
                                    emotion = emotion_map[emotion_code]
                                    src = os.path.join(root, file)
                                    dst = os.path.join(ravdess_processed, emotion, file)
                                    shutil.copy2(src, dst)
            
            return True
        
        except Exception as e:
            logger.error(f"Error preparing emotion datasets: {str(e)}")
            return False
    
    def prepare_all_datasets(self):
        """
        Prepare all datasets for use in the project.
        
        Returns:
            bool: True if successful, False otherwise
        """
        autism_success = self.prepare_autism_datasets()
        emotion_success = self.prepare_emotion_datasets()
        
        return autism_success and emotion_success
    
    def create_sample_data(self):
        """
        Create sample data for testing when real datasets aren't available.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import pandas as pd
            import numpy as np
            
            logger.info("Creating sample data for testing")
            
            # Create sample autism screening data
            sample_size = 100
            
            # Features for autism screening
            sample_data = {
                'A1': np.random.randint(0, 2, sample_size),  # 0: No, 1: Yes
                'A2': np.random.randint(0, 2, sample_size),
                'A3': np.random.randint(0, 2, sample_size),
                'A4': np.random.randint(0, 2, sample_size),
                'A5': np.random.randint(0, 2, sample_size),
                'A6': np.random.randint(0, 2, sample_size),
                'A7': np.random.randint(0, 2, sample_size),
                'A8': np.random.randint(0, 2, sample_size),
                'A9': np.random.randint(0, 2, sample_size),
                'A10': np.random.randint(0, 2, sample_size),
                'age': np.random.randint(2, 18, sample_size),  # Age in years
                'gender': np.random.choice(['m', 'f'], sample_size),
                'ethnicity': np.random.choice(['White', 'Asian', 'Black', 'Hispanic', 'Other'], sample_size),
                'jaundice': np.random.choice(['yes', 'no'], sample_size),
                'autism': np.random.choice(['yes', 'no'], sample_size),
                'country_of_res': np.random.choice(['United States', 'UK', 'India', 'China', 'Canada'], sample_size),
                'used_app_before': np.random.choice(['yes', 'no'], sample_size),
                'result': np.random.uniform(0, 10, sample_size),  # AQ-10 score
                'age_desc': np.random.choice(['2-5 years', '6-10 years', '11-17 years'], sample_size),
                'relation': np.random.choice(['Parent', 'Self', 'Health care professional', 'Relative'], sample_size),
                'Class/ASD': np.random.choice(['YES', 'NO'], sample_size)  # Target variable
            }
            
            sample_df = pd.DataFrame(sample_data)
            
            # Save sample data
            sample_path = os.path.join(self.data_dir, 'processed', 'sample_autism_child.csv')
            sample_df.to_csv(sample_path, index=False)
            
            logger.info(f"Sample autism screening data saved to {sample_path}")
            
            # Create behavioral feature samples (in JSON format)
            behavioral_samples = []
            
            for i in range(50):
                sample = {
                    'eye_contact': np.random.uniform(0, 1),
                    'responds_to_name': np.random.uniform(0, 1),
                    'social_smile': np.random.uniform(0, 1),
                    'interest_in_peers': np.random.uniform(0, 1),
                    'pointing': np.random.uniform(0, 1),
                    'showing': np.random.uniform(0, 1),
                    'pretend_play': np.random.uniform(0, 1),
                    'follows_gaze': np.random.uniform(0, 1),
                    'repetitive_behaviors': np.random.uniform(0, 1),
                    'unusual_interests': np.random.uniform(0, 1),
                    'gaze_fixation_duration': np.random.uniform(0, 1),
                    'gaze_transition_frequency': np.random.uniform(0, 1),
                    'face_looking_ratio': np.random.uniform(0, 1),
                    'joint_attention_score': np.random.uniform(0, 1),
                    'vocalization_frequency': np.random.uniform(0, 1),
                    'prosody_variation': np.random.uniform(0, 1),
                    'speech_rate': np.random.uniform(0, 1),
                    'articulation_quality': np.random.uniform(0, 1),
                    'autism': np.random.choice([0, 1], p=[0.7, 0.3])  # 30% autism for balance
                }
                behavioral_samples.append(sample)
            
            # Save behavioral samples
            import json
            behavioral_path = os.path.join(self.data_dir, 'processed', 'sample_behavioral_features.json')
            with open(behavioral_path, 'w') as f:
                json.dump(behavioral_samples, f)
            
            logger.info(f"Sample behavioral features saved to {behavioral_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            return False 