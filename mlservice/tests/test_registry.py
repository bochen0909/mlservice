"""
Tests for the route registry system.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlservice.core.registry import RouteRegistry, registry
from mlservice.main import setup_routes, app
from mlservice.demo.models import Item

def test_registry_singleton():
    """Test that RouteRegistry maintains singleton pattern."""
    r1 = RouteRegistry.get_instance()
    r2 = RouteRegistry.get_instance()
    assert r1 is r2
    
    # Should not be able to create new instance directly
    with pytest.raises(RuntimeError):
        RouteRegistry()

def test_route_registration():
    """Test basic route registration functionality."""
    test_app = FastAPI()
    test_registry = RouteRegistry.get_instance()
    
    @test_registry.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    test_registry.apply_routes(test_app)
    client = TestClient(test_app)
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "test"}

def test_multiple_methods():
    """Test registration of multiple HTTP methods."""
    test_app = FastAPI()
    test_registry = RouteRegistry.get_instance()
    
    test_data = {"key": "value"}
    
    @test_registry.get("/data")
    async def get_data():
        return test_data
    
    @test_registry.post("/data")
    async def post_data(data: dict):
        test_data.update(data)
        return test_data
    
    test_registry.apply_routes(test_app)
    client = TestClient(test_app)
    
    # Test GET
    response = client.get("/data")
    assert response.status_code == 200
    assert response.json() == test_data
    
    # Test POST
    new_data = {"new_key": "new_value"}
    response = client.post("/data", json=new_data)
    assert response.status_code == 200
    assert response.json() == {**test_data, **new_data}

@pytest.fixture
def client():
    """Create a test client with all routes registered."""
    setup_routes()  # This will import demo routes
    return TestClient(app)

def test_demo_routes(client):
    """Test demo route endpoints."""
    # Test basic demo endpoint
    response = client.get("/demo")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from registered demo endpoint!"}
    
    # Test items endpoints
    item_data = Item(name="test item").model_dump()
    
    # Create item
    response = client.post("/demo/items", json=item_data)
    assert response.status_code == 200
    assert "Created item" in response.json()["message"]
    
    # Get item
    response = client.get("/demo/items/1")
    assert response.status_code == 200
    assert response.json()["item_id"] == 1
    
    # Get item with details
    response = client.get("/demo/items/1?detail=true")
    assert response.status_code == 200
    assert "extra" in response.json()
    
    # Update item
    response = client.put("/demo/items/1", json=item_data)
    assert response.status_code == 200
    assert response.json()["message"] == "Updated item 1"
    
    # Delete item
    response = client.delete("/demo/items/1")
    assert response.status_code == 200
    assert response.json()["message"] == "Deleted item 1"

def test_group_endpoints(client):
    """Test grouped endpoints."""
    # Test base group endpoint
    response = client.get("/demo/group")
    assert response.status_code == 200
    assert response.json()["message"] == "Group base endpoint"
    
    # Test sub group endpoint
    response = client.get("/demo/group/subpath")
    assert response.status_code == 200
    assert response.json()["message"] == "Group sub endpoint"

def test_external_routes(client):
    """Test external route endpoints."""
    # Test basic external endpoint
    response = client.get("/external")
    assert response.status_code == 200
    assert response.json()["message"] == "Hello from external route!"
    
    # Test external data endpoint
    response = client.get("/external/data")
    assert response.status_code == 200
    assert response.json()["source"] == "external module"
    assert "data" in response.json()
