from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
import os
import shutil
from mlservice.core.registry import registry

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

# Register the router
registry.register_router(router)
