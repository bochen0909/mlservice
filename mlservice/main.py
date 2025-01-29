"""
Main FastAPI application module.
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Run the ML Service API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
