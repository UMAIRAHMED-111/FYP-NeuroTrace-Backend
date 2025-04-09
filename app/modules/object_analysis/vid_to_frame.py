import cv2
import os

def extract_frames(video_path, interval_sec=1):
    # Define relative output directory
    output_dir = os.path.join("assets", "frames")
    os.makedirs(output_dir, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"Video duration: {duration_sec:.2f}s, FPS: {fps}, Total Frames: {total_frames}")

    count = 0
    current_sec = 0

    while current_sec < duration_sec:
        frame_number = int(current_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved {frame_path}")
            count += 1
        else:
            print(f"Warning: Could not read frame at {current_sec} seconds")
        current_sec += interval_sec

    cap.release()
    print("Done extracting frames.")

if __name__ == "__main__":
    # Replace this with your actual video file path (can be relative or absolute)
    video_path = "F:\\Anfa Backup\\E\\IBA_8th_sm\\FYP\\brom_14\\borm\\assets\\videos\\2110972-uhd_3840_2160_30fps.mp4"  # <-- update this
    extract_frames(video_path)
