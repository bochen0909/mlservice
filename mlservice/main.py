"""
Main FastAPI application module.

This module provides the main FastAPI application with Swagger UI documentation 
and dynamic route registration capabilities. The API documentation is available 
at /docs endpoint.
"""

import argparse
from pathlib import Path
import uvicorn
from fastapi import FastAPI

from mlservice.core.registry import registry
# Import all models to make them visible in Swagger UI
from mlservice.demo import models, external_models

app = FastAPI(
    title="ML Service",
    description="""
    Machine Learning Service API with dynamic route registration.
    
    Features:
    - Dynamic route registration via registry pattern
    - Support for external route integration
    - Automatic Swagger/OpenAPI documentation
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": 1}
)

@app.get("/", 
         tags=["General"],
         summary="Root endpoint",
         response_description="Welcome message object")
async def hello():
    """
    Root endpoint returning a welcome message.
    
    Returns:
        dict: A JSON object containing a welcome message
        
    Example response:
        {
            "message": "Hello World"
        }
    """
    return {"message": "Hello World"}

def setup_routes():
    """Setup all registered routes and import external routes."""
    # Import demo routes
    import mlservice.demo.routes  # noqa

    # Import external routes as a Python module
    try:
        registry.import_routes_from_module("external_routes")
    except ValueError as e:
        print(f"Warning: Failed to import external routes: {e}")

    # Apply all registered routes to the FastAPI app
    registry.apply_routes(app)

def main():
    """Run the FastAPI application."""
    setup_routes()
    parser = argparse.ArgumentParser(description="Run the ML Service API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
