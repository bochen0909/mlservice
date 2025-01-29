"""
Main FastAPI application module.
"""

import argparse
from pathlib import Path
import uvicorn
from fastapi import FastAPI

from mlservice.core.registry import registry

app = FastAPI(
    title="ML Service",
    description="Machine Learning Service API with dynamic route registration",
    version="0.1.0"
)

@app.get("/")
async def hello():
    """
    Root endpoint returning a welcome message.
    """
    return {"message": "Hello World"}

def setup_routes():
    """Setup all registered routes and import external routes."""
    # Import demo routes
    import mlservice.demo.routes  # noqa

    # Example of importing routes from external path
    external_routes = Path(__file__).parent.parent / "external_routes"
    if external_routes.exists():
        registry.import_routes_from_path(str(external_routes))

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
