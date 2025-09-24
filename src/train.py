import os
from ultralytics import YOLO

def main():
    # Load base YOLO model (you can switch yolov8n.pt → yolov8s.pt for better accuracy)
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="../data/weapons_dataset/data.yaml",  # path to your Roboflow data.yaml
        epochs=50,                         # increase if dataset is big
        imgsz=640,                         # standard image size
        batch=16,                          # adjust based on your GPU/CPU memory
        name="weapons_train",              # result folder name under /results
        project="../results",              # save all runs inside results/
        patience=10                        # stop early if no improvement
    )

    print("✅ Training finished! Check results folder for weights and logs.")

if __name__ == "__main__":
    main()
