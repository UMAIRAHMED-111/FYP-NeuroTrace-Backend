from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import uuid
import shutil

from app.modules.object_analysis.main import analyze_video
from app.modules.embedding_model.DataIngestionPipeline import main_audio, main_text
from app.modules.embedding_model.Query import query_vector_store

router = APIRouter()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze-video")
async def analyze_video_route(
    video: UploadFile = File(...),
    user_id: str = Form(...)
):
    try:
        # Save uploaded video temporarily
        ext = os.path.splitext(video.filename)[1]
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)

        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Run analysis
        object_dict, scene, output_path = analyze_video(temp_filepath)
        
        # Call vector embedding model
        main_text(user_id, output_path)

        # Clean up
        os.remove(temp_filepath)

        return JSONResponse(content={
            "objects": object_dict,
            "scene": scene  
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    
@router.post("/analyze-audio")
async def analyze_audio_route(
    audio: UploadFile = File(...),
    user_id: str = Form(...)
):
    try:
        # Ensure it's a WAV file
        ext = os.path.splitext(audio.filename)[1]
        if ext.lower() != ".wav":
            return JSONResponse(status_code=400, content={"error": "Only WAV files are supported."})

        # Save uploaded audio temporarily
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)

        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        print(f"📄 Saved uploaded file to: {temp_filepath}")

        # Run the audio ingestion + processing pipeline
        main_audio(user_id, temp_filepath)

        # Clean up temporary file
        os.remove(temp_filepath)

        return JSONResponse(content={"message": "Audio processed successfully."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/query-vector")
async def query_vector_route(
    user_id: str = Form(...),
    query: str = Form(...),
    top_k: int = Form(5)
):
    try:
        results = query_vector_store(user_id, query, top_k=top_k)

        return JSONResponse(content={
            "results": [
                {
                    "text": r["text"],
                    "timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                }
                for r in results
            ]
        })

    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Vector store not found for this user."})
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})