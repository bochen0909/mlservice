"""
Demo routes to showcase the route registry functionality.
"""
from typing import Any, Dict, List
from fastapi import Path, Query
from mlservice.core.registry import registry

# Basic GET endpoint
@registry.get("/demo")
async def demo_endpoint() -> Dict[str, str]:
    """Simple demonstration endpoint."""
    return {"message": "Hello from registered demo endpoint!"}

# POST endpoint with JSON body
@registry.post("/demo/items")
async def create_item(item: Dict[str, str]) -> Dict[str, str]:
    """Demo POST endpoint that accepts JSON data."""
    return {"message": f"Created item: {item}"}

# GET with path and query parameters
@registry.get("/demo/items/{item_id}")
async def get_item(
    item_id: int = Path(..., description="The ID of the item"),
    detail: bool = Query(False, description="Include extra details"),
) -> Dict[str, Any]:
    """Demo endpoint with path and query parameters."""
    response = {"item_id": item_id}
    if detail:
        response["extra"] = "Detailed information here"
    return response

# PUT endpoint
@registry.put("/demo/items/{item_id}")
async def update_item(
    item_id: int,
    item: Dict[str, str]
) -> Dict[str, str]:
    """Demo PUT endpoint for updating items."""
    return {
        "message": f"Updated item {item_id}",
        "data": item
    }

# DELETE endpoint
@registry.delete("/demo/items/{item_id}")
async def delete_item(item_id: int) -> Dict[str, str]:
    """Demo DELETE endpoint."""
    return {"message": f"Deleted item {item_id}"}

# Example of grouping related endpoints
@registry.get("/demo/group")
async def group_base() -> Dict[str, str]:
    """Base endpoint for a group of related endpoints."""
    return {"message": "Group base endpoint"}

@registry.get("/demo/group/subpath")
async def group_sub() -> Dict[str, str]:
    """Sub-endpoint in the group."""
    return {"message": "Group sub endpoint"}
