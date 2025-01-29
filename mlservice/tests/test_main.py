"""
Tests for the main FastAPI application.
"""
from fastapi.testclient import TestClient
from mlservice.main import app, setup_routes

def test_root_endpoint():
    """Test the root endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_setup_routes():
    """Test route setup and integration."""
    setup_routes(['external_routes', 'external_routes.demo'])
    client = TestClient(app)
    
    # Test external routes are registered if available
    response = client.get("/external")
    assert response.status_code == 200
