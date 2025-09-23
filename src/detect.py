from ultralytics import YOLO
import sys
import os

# Paths
MODEL_PATH = os.path.join("models", "yolov8n.pt")
RESULTS_DIR = "results"

# Load YOLO model
model = YOLO(MODEL_PATH)

def detect_image(image_path):
    results = model(image_path, save=True, project=RESULTS_DIR, name="images")
    print(f"Results saved in {RESULTS_DIR}/images")

def detect_video(video_path):
    results = model(video_path, save=True, project=RESULTS_DIR, name="videos")
    print(f"Results saved in {RESULTS_DIR}/videos")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/detect.py [image|video] path_to_file")
        sys.exit(1)

    mode, path = sys.argv[1], sys.argv[2]

    if mode == "image":
        detect_image(path)
    elif mode == "video":
        detect_video(path)
    else:
        print("Invalid mode! Use 'image' or 'video'.")
