"""
Pydantic models for request and response data.
"""
from typing import Dict, Optional
from pydantic import BaseModel

class Item(BaseModel):
    """Item model for request/response data."""
    name: str
    description: Optional[str] = None

class ItemResponse(BaseModel):
    """Response model for item operations."""
    message: str
    data: Optional[Dict] = None

class ItemDetail(BaseModel):
    """Model for detailed item response."""
    item_id: int
    extra: Optional[str] = None
