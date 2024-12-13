from ultralytics import YOLO

def train_model():
    model = YOLO(r"D:\Fracture Detection\models\yolo11l-seg.pt")
    results = model.train(
        data = r"D:\Fracture Detection\config\data.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 1,
        project=r"D:\Fracture Detection\weights",
        name="yolo11m_segmentation"
    )


if __name__ == "__main__":
    train_model()