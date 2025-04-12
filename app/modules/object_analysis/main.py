from app.modules.object_analysis.vid_to_frame import extract_frames
from app.modules.object_analysis.frames_pipeline import FramePipeline

def analyze_video(video_path: str, interval: int = 1):
    """
    Extract frames from the given video, run object and scene detection,
    and return a summary of detected objects and the majority scene.

    Args:
        video_path (str): Path to the video file.
        interval (int): Time interval (in seconds) between frames to extract.

    Returns:
        Tuple[Dict[str, int], str]: A dictionary of object counts and the final scene prediction.
    """
    print("\n🎬 Step 1: Extracting frames from video...")
    extract_frames(video_path, interval_sec=interval)

    print("\n🧠 Step 2: Running frame processing pipeline...")
    pipeline = FramePipeline()
    object_dict, final_scene, output_path  = pipeline.process_frames()

    print("\n📦 Final Output:")
    print(f"Detected Objects: {object_dict}")
    print(f"Majority Scene Prediction: {final_scene}")
    print(f"Output Path: {output_path}")

    return object_dict, final_scene, output_path
