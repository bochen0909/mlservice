"""
Main FastAPI application module.
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="ML Service", description="Machine Learning Service API", version="0.1.0"
)

@app.get("/")
async def hello():
    """
    Root endpoint returning a welcome message.
    """
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
