import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime
from mlservice.main import app

client = TestClient(app)

def test_upload_file_success(tmp_path):
    # Setup
    test_file_content = b"test content"
    mock_time = datetime(2025, 1, 1, 12, 0, 0)
    
    with patch.dict(os.environ, {"ML_HOME": str(tmp_path)}), \
         patch("mlservice.core.upload_routes.datetime") as mock_datetime:
        
        mock_datetime.now.return_value = mock_time
        
        # Create test file
        files = {"file": ("test.txt", test_file_content, "text/plain")}
        
        # Execute
        response = client.post("/upload", files=files)
        
        # Assert
        assert response.status_code == 200
        assert "File uploaded successfully" in response.json()["message"]
        
        # Verify file was saved correctly
        expected_path = tmp_path / "data/2025/01/01/12/00/00/test.txt"
        assert expected_path.exists()
        assert expected_path.read_bytes() == test_file_content

def test_upload_file_missing_ml_home():
    # Setup
    test_file_content = b"test content"
    files = {"file": ("test.txt", test_file_content, "text/plain")}
    
    with patch.dict(os.environ, {}, clear=True):
        # Execute
        response = client.post("/upload", files=files)
        
        # Assert
        assert response.status_code == 500
        assert "ML_HOME environment variable not set" in response.json()["detail"]

def test_upload_file_filesystem_error(tmp_path):
    # Setup
    test_file_content = b"test content"
    mock_time = datetime(2025, 1, 1, 12, 0, 0)
    
    with patch.dict(os.environ, {"ML_HOME": str(tmp_path)}), \
         patch("mlservice.core.upload_routes.datetime") as mock_datetime, \
         patch("mlservice.core.upload_routes.os.makedirs") as mock_makedirs:
        
        mock_datetime.now.return_value = mock_time
        mock_makedirs.side_effect = OSError("Permission denied")
        
        files = {"file": ("test.txt", test_file_content, "text/plain")}
        
        # Execute
        response = client.post("/upload", files=files)
        
        # Assert
        assert response.status_code == 500
        assert "Permission denied" in response.json()["detail"]

def test_upload_file_invalid_file():
    # Setup
    with patch.dict(os.environ, {"ML_HOME": "/tmp"}):
        # Execute
        response = client.post("/upload")
        
        # Assert
        assert response.status_code == 422  # FastAPI validation error
