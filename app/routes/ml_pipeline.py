from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import uuid
import shutil

from app.modules.object_analysis.main import analyze_video

router = APIRouter()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")
async def analyze_video_route(video: UploadFile = File(...)):
    try:
        # Save uploaded video temporarily
        ext = os.path.splitext(video.filename)[1]
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)

        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Run analysis
        object_dict, scene = analyze_video(temp_filepath)

        # Clean up
        os.remove(temp_filepath)

        return JSONResponse(content={
            "objects": object_dict,
            "scene": scene
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
