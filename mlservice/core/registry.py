"""
Core route registry functionality.
"""
from typing import Any, Callable, Dict, List
import importlib.util
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

    def import_routes_from_module(self, module_name: str) -> None:
        """
        Import routes from a Python module and its submodules.
        
        Args:
            module_name: Full Python module name (e.g. 'external_routes')
        """
        try:
            module = importlib.import_module(module_name)
            # Get the module directory to scan for submodules
            module_path = Path(module.__file__).parent
            
            # Import all non-private Python files in the module directory
            for file in module_path.rglob("*.py"):
                if file.name.startswith("_"):
                    continue
                
                # Calculate relative path from module root to build full module name
                relative_path = file.relative_to(module_path)
                submodule_parts = list(relative_path.parent.parts)
                if submodule_parts == ["."]:
                    submodule_parts = []
                    
                submodule_name = ".".join(
                    [module_name] + submodule_parts + [file.stem]
                )
                
                if submodule_name != module_name:  # Skip the root module
                    importlib.import_module(submodule_name)
        except ImportError as e:
            raise ValueError(f"Could not import module {module_name}: {e}")

# Create the singleton instance
registry = RouteRegistry.get_instance()
