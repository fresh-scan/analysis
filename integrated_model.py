import sys
import json
import torch
import cv2
import easyocr
import pathlib
import re
import warnings
from pathlib import Path

# 불필요한 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)

# Global 변수
ingredient_model = None
label_model = None
reader = None

# 라벨 여부 판별 기준
label_free_ingredients = {"tomato", "pimang", "carrot", "gaji", "cabagge", "beef", "fish"}
labeled_ingredients = {"tofu", "sauce", "milk"}


def initialize_yolo_model(model_path, conf_threshold=0.03, yolov5_dir="", model_type="ingredient"):
    global ingredient_model, label_model
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    model = torch.hub.load(yolov5_dir, "custom", path=model_path, source="local")
    model.conf = conf_threshold
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "ingredient":
        ingredient_model = model
    elif model_type == "label":
        label_model = model

    pathlib.PosixPath = temp


def initialize_easyocr():
    global reader
    reader = easyocr.Reader(["ko", "en"], gpu=torch.cuda.is_available())
    print("EasyOCR initialized.", file=sys.stderr)


def extract_date(text):
    date_pattern = r"\d{4}\.\d{2}\.\d{2}"
    matches = re.findall(date_pattern, text)
    return matches[0] if matches else None


def process_images(image_paths):
    if ingredient_model is None or label_model is None or reader is None:
        raise ValueError("Models or OCR reader are not initialized.")

    results = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        ingredient_results = ingredient_model(image_path)
        detections = ingredient_results.xyxy[0].cpu().numpy()

        detected_labels = []
        expiration_dates = []

        if detections.size > 0:
            for box in detections:
                x1, y1, x2, y2, conf, cls = map(int, box[:6])
                label = ingredient_model.names[cls]
                detected_labels.append(label)

                if label in {"tofu", "sauce", "milk"}:  # 라벨이 필요한 재료
                    cropped_region = image[y1:y2, x1:x2]
                    label_results = label_model(cropped_region)
                    label_detections = label_results.xyxy[0].cpu().numpy()

                    for label_box in label_detections:
                        lx1, ly1, lx2, ly2, lconf, lcls = map(int, label_box[:6])
                        label_region = cropped_region[ly1:ly2, lx1:lx2]
                        text_results = reader.readtext(label_region)
                        text = " ".join([res[1] for res in text_results]).strip()
                        date = extract_date(text)
                        if date:
                            expiration_dates.append(date)

        text_results = reader.readtext(image)
        for _, text, _ in text_results:
            date = extract_date(text)
            if date:
                expiration_dates.append(date)

        results.append({
            "detected_labels": detected_labels,
            "expiration_dates": expiration_dates,
        })

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image paths provided"}))
        sys.exit(1)

    if ingredient_model is None or label_model is None or reader is None:
        ingredient_model_path = "C:/fresh-Scan/analysis/models/ingredient_model.pt"
        label_model_path = "C:/fresh-Scan/analysis/models/label_model.pt"
        yolov5_dir = "C:/HCI_freshScan/ultralytics/yolov5"
        initialize_yolo_model(ingredient_model_path, yolov5_dir=yolov5_dir, model_type="ingredient")
        initialize_yolo_model(label_model_path, yolov5_dir=yolov5_dir, model_type="label")
        initialize_easyocr()

    image_paths = sys.argv[1:]
    try:
        results = process_images(image_paths)
        print(json.dumps(results))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
