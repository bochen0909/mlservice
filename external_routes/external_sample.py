"""
Example of external routes that can be imported into the ML service.
"""
from typing import Dict
from mlservice.core.registry import registry

@registry.get("/external")
async def external_endpoint() -> Dict[str, str]:
    """Example endpoint from external module."""
    return {"message": "Hello from external route!"}

@registry.get("/external/data")
async def external_data() -> Dict[str, str]:
    """Example data endpoint from external module."""
    return {
        "source": "external module",
        "data": {
            "key": "value",
            "status": "active"
        }
    }
