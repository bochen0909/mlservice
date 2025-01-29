"""
Core route registry functionality.
"""
from typing import Any, Callable, Dict, List, Optional, Type
import importlib.util
import os
from pathlib import Path
from fastapi import FastAPI, APIRouter

class RouteRegistry:
    """Registry for managing API routes across multiple modules and projects."""
    
    _instance = None
    _routes: List[Dict[str, Any]] = []
    
    def __init__(self):
        """Initialize the route registry."""
        if RouteRegistry._instance is not None:
            raise RuntimeError("RouteRegistry is a singleton")
        RouteRegistry._instance = self
    
    @classmethod
    def get_instance(cls) -> 'RouteRegistry':
        """Get the singleton instance of RouteRegistry."""
        if cls._instance is None:
            cls._instance = RouteRegistry()
        return cls._instance
    
    @classmethod
    def register_endpoint(
        cls,
        path: str,
        methods: List[str],
        **kwargs
    ) -> Callable:
        """
        Decorator for registering route handlers.
        
        Args:
            path: The URL path for the endpoint
            methods: List of HTTP methods (GET, POST, etc.)
            **kwargs: Additional FastAPI route parameters
        """
        def decorator(func: Callable) -> Callable:
            registry = cls.get_instance()
            registry._routes.append({
                'path': path,
                'methods': methods,
                'handler': func,
                'kwargs': kwargs
            })
            return func
        return decorator

    @classmethod
    def get(cls, path: str, **kwargs) -> Callable:
        """Decorator for registering GET endpoints."""
        return cls.register_endpoint(path, ['GET'], **kwargs)

    @classmethod
    def post(cls, path: str, **kwargs) -> Callable:
        """Decorator for registering POST endpoints."""
        return cls.register_endpoint(path, ['POST'], **kwargs)

    @classmethod
    def put(cls, path: str, **kwargs) -> Callable:
        """Decorator for registering PUT endpoints."""
        return cls.register_endpoint(path, ['PUT'], **kwargs)

    @classmethod
    def delete(cls, path: str, **kwargs) -> Callable:
        """Decorator for registering DELETE endpoints."""
        return cls.register_endpoint(path, ['DELETE'], **kwargs)

    def apply_routes(self, app: FastAPI) -> None:
        """
        Apply all registered routes to a FastAPI application.
        
        Args:
            app: FastAPI application instance
        """
        router = APIRouter()
        for route in self._routes:
            for method in route['methods']:
                endpoint = getattr(router, method.lower())
                endpoint(route['path'], **route['kwargs'])(route['handler'])
        app.include_router(router)

    def import_routes_from_path(self, path: str) -> None:
        """
        Import routes from Python files in the specified path.
        
        Args:
            path: Directory path containing route definitions
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        for file in path.rglob("*.py"):
            if file.name.startswith("_"):
                continue
            
            module_name = f"mlservice.imported_routes.{file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

# Create the singleton instance
registry = RouteRegistry.get_instance()
