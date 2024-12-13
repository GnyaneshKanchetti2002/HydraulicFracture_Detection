import os
from ultralytics import YOLO

def validate(model_path = r"D:\Fracture Detection\weights\yolo11m_segmentation\weights\best.pt",
             data_path = r"D:\Fracture Detection\config\data.yaml"):
    model = YOLO(model = model_path)
    model.val(data = data_path)

    return None

def testing(model_path = r"D:\Fracture Detection\weights\yolo11m_segmentation(with adv augmentation)\weights\best.pt",
            test_img_path = r"D:\Fracture Detection\data\images\test",
            output_path = r"D:\Fracture Detection\output"):
    model = YOLO(model = model_path)

    os.makedirs(output_path, exist_ok = True)

    results = model.predict(
        source = test_img_path,
        conf = 0.5,
        save = True,
        project = output_path,
        name = 'test1(with adv augmentation 0.5)',
        imgsz = 640
    )

    print(f"Predicted images saved to {output_path}")

    return None





if __name__ == "__main__":
    testing()