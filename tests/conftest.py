import os
import pytest
import tempfile
from src.app import create_app

@pytest.fixture(scope='session')
def app():
    """Create and configure a new app instance for the test session."""
    # Create a temporary directory for uploads
    with tempfile.TemporaryDirectory() as temp_dir:
        app = create_app()
        app.config.update({
            'TESTING': True,
            'UPLOAD_FOLDER': temp_dir,
            'WTF_CSRF_ENABLED': False
        })
        yield app

@pytest.fixture(scope='session')
def client(app):
    """Create a test client for the app."""
    return app.test_client()

@pytest.fixture(scope='session')
def runner(app):
    """Create a test runner for the app's CLI commands."""
    return app.test_cli_runner()

@pytest.fixture(scope='session')
def sample_image():
    """Provide a path to a sample image for testing."""
    return os.path.join('tests', 'test_data', 'test_image.jpg')

@pytest.fixture(scope='session')
def invalid_file():
    """Provide a path to an invalid file for testing."""
    return os.path.join('tests', 'test_data', 'invalid.txt')

@pytest.fixture(scope='session')
def sample_screening_data():
    """Provide sample screening questionnaire data for testing."""
    return {
        'question1': 'yes',
        'question2': 'no',
        'question3': 'sometimes',
        'question4': 'rarely',
        'question5': 'yes',
        'question6': 'no',
        'question7': 'sometimes',
        'question8': 'yes',
        'question9': 'no',
        'question10': 'rarely'
    } 