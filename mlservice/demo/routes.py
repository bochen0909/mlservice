"""
Demo routes to showcase the route registry functionality.
"""
from typing import Dict
from fastapi import Path, Query
from mlservice.core.registry import registry
from mlservice.demo.models import Item, ItemResponse, ItemDetail

# Basic GET endpoint
@registry.get("/demo", response_model=Dict[str, str])
async def demo_endpoint():
    """Simple demonstration endpoint."""
    return {"message": "Hello from registered demo endpoint!"}

# POST endpoint with JSON body
@registry.post("/demo/items", response_model=ItemResponse)
async def create_item(item: Item):
    """Demo POST endpoint that accepts JSON data."""
    return ItemResponse(
        message=f"Created item: {item.model_dump()}",
        data=item.model_dump()
    )

# GET with path and query parameters
@registry.get("/demo/items/{item_id}", response_model=ItemDetail)
async def get_item(
    item_id: int = Path(..., description="The ID of the item"),
    detail: bool = Query(False, description="Include extra details"),
):
    """Demo endpoint with path and query parameters."""
    response = ItemDetail(item_id=item_id)
    if detail:
        response.extra = "Detailed information here"
    return response

# PUT endpoint
@registry.put("/demo/items/{item_id}", response_model=ItemResponse)
async def update_item(
    item_id: int,
    item: Item
) -> ItemResponse:
    """Demo PUT endpoint for updating items."""
    return ItemResponse(
        message=f"Updated item {item_id}",
        data=item.model_dump()
    )

# DELETE endpoint
@registry.delete("/demo/items/{item_id}", response_model=ItemResponse)
async def delete_item(item_id: int) -> ItemResponse:
    """Demo DELETE endpoint."""
    return ItemResponse(message=f"Deleted item {item_id}")

# Example of grouping related endpoints
@registry.get("/demo/group", response_model=Dict[str, str])
async def group_base():
    """Base endpoint for a group of related endpoints."""
    return {"message": "Group base endpoint"}

@registry.get("/demo/group/subpath", response_model=Dict[str, str])
async def group_sub():
    """Sub-endpoint in the group."""
    return {"message": "Group sub endpoint"}
