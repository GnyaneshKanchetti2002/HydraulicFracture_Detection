from ultralytics import YOLO

def retrain_model():
    model = YOLO(r"D:\Fracture Detection\weights\yolo11m_segmentation(with augmentation)\weights\best.pt")
    results = model.train(
        data = r"D:\Fracture Detection\config\data.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 1,
        flipud=0.45,  # Vertical flip probability
        fliplr=0.45,  # Horizontal flip probability
        degrees=30,  # Random rotation degrees
        scale=1.0,  # Scale range, e.g., 0.5 means [0.5, 1.5] of original size
        hsv_s = 0.5,
        mosaic = 1.0,  # image mosaic (probability)
        mixup = 0.2,  # image mixup (probability)
        copy_paste = 0.1,
        project=r"D:\Fracture Detection\weights",
        name="yolo11m_segmentation(with adv augmentation)"
    )


if __name__ == "__main__":
    retrain_model()