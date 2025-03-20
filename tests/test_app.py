import os
import pytest
from src.app import create_app
from werkzeug.datastructures import FileStorage

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app = create_app()
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = 'tests/uploads'
    return app

@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return app.test_client()

def test_home_page(client):
    """Test that the home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Autism Detection & Emotion Analysis' in response.data

def test_screening_page(client):
    """Test that the screening page loads successfully."""
    response = client.get('/screening')
    assert response.status_code == 200
    assert b'Autism Screening' in response.data

def test_emotion_page(client):
    """Test that the emotion analysis page loads successfully."""
    response = client.get('/emotion')
    assert response.status_code == 200
    assert b'Emotion Analysis' in response.data

def test_about_page(client):
    """Test that the about page loads successfully."""
    response = client.get('/about')
    assert response.status_code == 200
    assert b'About Our Project' in response.data

def test_404_error(client):
    """Test that 404 errors are handled correctly."""
    response = client.get('/nonexistent')
    assert response.status_code == 404
    assert b'Page Not Found' in response.data

def test_emotion_analysis_endpoint(client):
    """Test the emotion analysis API endpoint."""
    # Create a test image file
    test_file = FileStorage(
        stream=open('tests/test_data/test_image.jpg', 'rb'),
        filename='test_image.jpg',
        content_type='image/jpeg'
    )
    
    response = client.post(
        '/api/v1/analyze/emotion',
        data={'file': test_file},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'emotions' in data
    assert 'dominant_emotion' in data

def test_screening_analysis_endpoint(client):
    """Test the screening analysis API endpoint."""
    test_data = {
        'question1': 'yes',
        'question2': 'no',
        'question3': 'sometimes',
        # Add more test questions as needed
    }
    
    response = client.post(
        '/api/v1/analyze/screening',
        json=test_data
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'screening_result' in data
    assert 'recommendations' in data

def test_invalid_file_upload(client):
    """Test that invalid file uploads are rejected."""
    # Create an invalid file
    test_file = FileStorage(
        stream=open('tests/test_data/invalid.txt', 'rb'),
        filename='invalid.txt',
        content_type='text/plain'
    )
    
    response = client.post(
        '/api/v1/analyze/emotion',
        data={'file': test_file},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    assert b'Invalid file type' in response.data

def test_rate_limiting(client):
    """Test that rate limiting is working."""
    for _ in range(101):  # Make 101 requests
        client.get('/')
    
    response = client.get('/')
    assert response.status_code == 429  # Too Many Requests 