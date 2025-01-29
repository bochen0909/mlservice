"""
Example of external routes that can be imported into the ML service.
"""
from typing import Dict
from mlservice.core.registry import registry
from mlservice.demo.external_models import ExternalDataResponse

@registry.get("/external", response_model=Dict[str, str])
async def external_endpoint():
    """Example endpoint from external module."""
    return {"message": "Hello from external route!"}

@registry.get("/external/data", response_model=ExternalDataResponse)
async def external_data():
    """Example data endpoint from external module."""
    return ExternalDataResponse(
        source="external module",
        data={
            "key": "value",
            "status": "active"
        }
    )
