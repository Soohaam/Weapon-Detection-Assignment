import sys
import os
from ultralytics import YOLO

def run_inference(weights_path, source):
    # Load trained model
    model = YOLO(weights_path)

    # Root-level results folder
    base_results_dir = os.path.join("..", "results", "weapons_detection_results")
    os.makedirs(base_results_dir, exist_ok=True)

    # Check if source is image or video
    if source.lower().endswith((".jpg", ".jpeg", ".png")):
        # Save image output directly inside weapons_detection_results
        results = model.predict(
            source=source,
            imgsz=640,
            conf=0.25,
            save=True,
            project=base_results_dir,   # ✅ root-level results
            name=".",                   # ⚡ avoid "predict" folder
            exist_ok=True,
            show=True
        )
        print(f"✅ Image processed! Check {base_results_dir}/ for outputs.")

    elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        # Create folder with video filename (without extension)
        video_name = os.path.splitext(os.path.basename(source))[0]
        video_dir = os.path.join(base_results_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)

        # Save detections inside video folder
        results = model.predict(
            source=source,
            imgsz=640,
            conf=0.25,
            save=True,
            project=video_dir,          # ✅ save directly inside the video folder
            name=".",                   # ⚡ avoid "predict" subfolder
            exist_ok=True,
            show=True
        )
        print(f"🎥 Video processed! Frames + video saved in {video_dir}/")

    else:
        print("❌ Unsupported file format! Please provide an image (.jpg/.png) or video (.mp4/.avi/.mov/.mkv).")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/detect.py <path_to_weights> <source>")
        print("Example: python src/detect.py models/yolov8n.pt data/images/sample.jpg")
        print("Example: python src/detect.py models/yolov8n.pt data/videos/demo.mp4")
        sys.exit(1)

    weights = sys.argv[1]
    source = sys.argv[2]
    run_inference(weights, source)
