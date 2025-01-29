"""
Test cases for main FastAPI application.
"""

from fastapi.testclient import TestClient
from mlservice.main import app

client = TestClient(app)

def test_hello():
    """
    Test hello endpoint returns correct message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
