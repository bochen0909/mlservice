from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime
import os
import shutil

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

@router.get("/download/{year}/{month}/{day}/{hour}/{minute}/{second}/{filename}", tags=["File Download"])
async def download_file(year: str, month: str, day: str, hour: str, minute: str, second: str, filename: str):
    try:
        # Get the ML_HOME environment variable
        ml_home = os.environ.get("ML_HOME")
        if not ml_home:
            raise HTTPException(status_code=500, detail="ML_HOME environment variable not set")

        # Construct the file path
        file_path = os.path.join(ml_home, "data", year, month, day, hour, minute, second, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(file_path, filename=filename)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
