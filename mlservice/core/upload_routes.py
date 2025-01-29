from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime
import os
import shutil
from pathlib import Path

router = APIRouter()

@router.post("/upload", tags=["File Upload"])
async def upload_file(file: UploadFile = File(...)):
    try:
        # Get the ML_HOME environment variable
        ml_home = os.environ.get("ML_HOME")
        if not ml_home:
            raise HTTPException(status_code=500, detail="ML_HOME environment variable not set")

        # Generate time-structured path
        now = datetime.now()
        time_path = now.strftime("%Y/%m/%d/%H/%M/%S")
        
        # Create the full path
        full_path = os.path.join(ml_home, "data", time_path)
        os.makedirs(full_path, exist_ok=True)

        # Save the file
        file_path = os.path.join(full_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"message": "File uploaded successfully", "path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download", tags=["File Download"])
async def download_file(file_path: str = None):
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path parameter is required")
    
    # Get the ML_HOME environment variable
    ml_home = os.environ.get("ML_HOME")
    if not ml_home:
        raise HTTPException(status_code=500, detail="ML_HOME environment variable not set")

    # Convert paths to Path objects for better path manipulation
    ml_home_path = Path(ml_home).resolve()
    
    # Handle absolute and relative paths
    if file_path.startswith("/"):
        request_path = Path(file_path).resolve()
    else:
        # For relative paths, join with ML_HOME/data
        request_path = (ml_home_path / "data" / file_path).resolve()
    
    # Check if the resolved path is within ML_HOME
    if not request_path.is_relative_to(ml_home_path):
        raise HTTPException(status_code=400, detail="Access denied: Path is outside ML_HOME")
        
    try:
        # Convert back to string for compatibility with rest of the code
        full_path = str(request_path)
        # Check if file exists
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        # Get relative path from ML_HOME
        relative_path = os.path.relpath(full_path, ml_home)
        
        # Return both paths in headers and send file
        response = FileResponse(full_path, filename=os.path.basename(file_path))
        response.headers["X-Full-Path"] = full_path
        response.headers["X-Relative-Path"] = relative_path
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
