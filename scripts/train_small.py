from ultralytics import YOLO

def train_model():
    model = YOLO(r"D:\Fracture Detection\models\yolo11l-seg.pt")
    results = model.train(
        data = r"D:\Fracture Detection\config\data_small.yaml",
        epochs = 200,
        imgsz = 640,
        batch = 1,
        flipud=0.65,  # Vertical flip probability
        fliplr=0.65,  # Horizontal flip probability
        degrees=35,  # Random rotation degrees
        scale=0.9,  # Scale range, e.g., 0.5 means [0.5, 1.5] of original size
        hsv_s=0.9,
        mosaic=1.0,  # image mosaic (probability)
        mixup=0.5,  # image mixup (probability)
        copy_paste=0.3,
        project=r"D:\Fracture Detection\weights",
        name="yolo11m_segmentation (small frac)"
    )


if __name__ == "__main__":
    train_model()