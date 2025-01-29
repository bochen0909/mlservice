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

def test_download_file_relative_path_success(tmp_path):
    # Setup
    test_file_content = b"test content"
    file_path = os.path.join(str(tmp_path), "data", "test_dir")
    os.makedirs(file_path, exist_ok=True)
    test_file = os.path.join(file_path, "test.txt")
    with open(test_file, "wb") as f:
        f.write(test_file_content)
    
    with patch.dict(os.environ, {"ML_HOME": str(tmp_path)}):
        # Execute
        response = client.get("/download", params={"file_path": "test_dir/test.txt"})
        
        # Assert
        assert response.status_code == 200
        assert response.content == test_file_content
        assert response.headers["X-Full-Path"] == test_file
        assert response.headers["X-Relative-Path"] == os.path.join("data", "test_dir", "test.txt")

def test_download_file_absolute_path_success(tmp_path):
    # Setup
    test_file_content = b"test content"
    file_path = os.path.join(str(tmp_path), "data", "test_dir")
    os.makedirs(file_path, exist_ok=True)
    test_file = os.path.join(file_path, "test.txt")
    with open(test_file, "wb") as f:
        f.write(test_file_content)
    
    with patch.dict(os.environ, {"ML_HOME": str(tmp_path)}):
        # Execute
        response = client.get("/download", params={"file_path": test_file})
        
        # Assert
        assert response.status_code == 200
        assert response.content == test_file_content
        assert response.headers["X-Full-Path"] == test_file
        assert response.headers["X-Relative-Path"] == os.path.join("data", "test_dir", "test.txt")

def test_download_file_relative_ok(tmp_path):
    with patch.dict(os.environ, {"ML_HOME": str(tmp_path)}):
        # Try to access file with relative path
        response = client.get("/download", params={"file_path": "folder/test.txt"})
        
        # Assert - should return 404 since file doesn't exist
        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

def test_download_file_absolute_outside_ml_home(tmp_path):
    with patch.dict(os.environ, {"ML_HOME": str(tmp_path)}):
        # Try to access absolute path outside ML_HOME
        response = client.get("/download", params={"file_path": "/etc/passwd"})
        
        # Assert
        assert response.status_code == 400
        assert "Access denied: Path is outside ML_HOME" in response.json()["detail"]

def test_download_file_not_found(tmp_path):
    with patch.dict(os.environ, {"ML_HOME": str(tmp_path)}):
        # Execute
        response = client.get("/download", params={"file_path": "nonexistent.txt"})
        
        # Assert
        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

def test_download_file_missing_ml_home():
    with patch.dict(os.environ, {}, clear=True):
        # Execute
        response = client.get("/download", params={"file_path": "test.txt"})
        
        # Assert
        assert response.status_code == 500
        assert "ML_HOME environment variable not set" in response.json()["detail"]
