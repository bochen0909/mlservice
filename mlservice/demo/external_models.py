"""
Models for external route responses.
"""
from typing import Dict
from pydantic import BaseModel

class ExternalDataResponse(BaseModel):
    """Response model for external data endpoint."""
    source: str
    data: Dict[str, str]
