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

def setup_routes(module_names: list[str] | None = None):
    """
    Setup all registered routes and import external routes.
    
    Args:
        module_names (list[str] | None): List of external route module names to import.
            If None, no external routes are imported.
    """

    print(f"Setting up routes with module names: {module_names}")

    # Import external routes if provided
    if module_names:
        for module_name in module_names:
            try:
                print(f"Attempting to import routes from module: {module_name}")
                registry.import_routes_from_module(module_name)
                print(f"Successfully imported routes from module: {module_name}")
            except ValueError as e:
                print(f"Warning: Failed to import external routes from {module_name}: {e}")
            except Exception as e:
                print(f"Unexpected error importing routes from {module_name}: {str(e)}")

    # Apply all registered routes to the FastAPI app
    print("Applying registered routes to FastAPI app")
    registry.apply_routes(app)
    print("Finished applying routes")

def main():
    """Run the FastAPI application."""
    parser = argparse.ArgumentParser(description="Run the ML Service API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--external-routines", nargs="+", help="List of external routine modules to import")
    args = parser.parse_args()
    
    setup_routes(args.external_routines)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
