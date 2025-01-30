"""
Tests for the main FastAPI application.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from mlservice.main import app, setup_routes, main

def test_root_endpoint():
    """Test the root endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_setup_routes_with_valid_module():
    """Test route setup with valid external module."""
    setup_routes(['external_routes'])
    client = TestClient(app)
    
    # Test external routes are registered if available
    response = client.get("/external")
    assert response.status_code == 200

def test_setup_routes_with_no_modules():
    """Test route setup without any external modules."""
    setup_routes(None)
    client = TestClient(app)
    # Basic app functionality should still work
    response = client.get("/")
    assert response.status_code == 200

def test_setup_routes_with_invalid_module():
    """Test route setup with invalid module name."""
    setup_routes(['nonexistent_module'])
    client = TestClient(app)
    # Basic app functionality should still work
    response = client.get("/")
    assert response.status_code == 200

def test_setup_routes_with_multiple_modules():
    """Test route setup with multiple modules."""
    setup_routes(['external_routes', 'external_routes.demo'])
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200

@patch('argparse.ArgumentParser.parse_args')
@patch('uvicorn.run')
def test_main_default_args(mock_run, mock_parse_args):
    """Test main function with default arguments."""
    args = MagicMock()
    args.host = "0.0.0.0"
    args.port = 8000
    args.external_routines = None
    mock_parse_args.return_value = args
    
    main()
    
    mock_run.assert_called_once_with(app, host="0.0.0.0", port=8000)

@patch('argparse.ArgumentParser.parse_args')
@patch('uvicorn.run')
def test_main_custom_args(mock_run, mock_parse_args):
    """Test main function with custom arguments."""
    args = MagicMock()
    args.host = "localhost"
    args.port = 9000
    args.external_routines = ["external_routes"]
    mock_parse_args.return_value = args
    
    main()
    
    mock_run.assert_called_once_with(app, host="localhost", port=9000)
