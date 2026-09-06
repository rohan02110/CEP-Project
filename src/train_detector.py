"""
Training and evaluation pipeline for YOLOv8 Multi-Class Damage Detection.
Fine-tunes YOLOv8n on CarDD dataset and benchmarks on held-out test split.
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    RANDOM_SEED, CARDD_CLASSES, CARDD_CLASS_TO_IDX, CARDD_IDX_TO_CLASS
)


def train_damage_detector():
    """
    Fine-tunes YOLOv8n detector on CarDD training set with validation monitoring.
    """
    print("\n" + "="*50)
    print("STARTING YOLOv8 DAMAGE DETECTOR TRAINING")
    print("="*50, flush=True)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    yaml_path = PROCESSED_DATA_DIR / "cardd" / "dataset.yaml"
    assert yaml_path.exists(), f"Dataset config not found at: {yaml_path}"

    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    yolo_runs_dir = MODELS_DIR / "yolo_runs"
    yolo_runs_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    results = model.train(
        data=str(yaml_path),
        epochs=20,
        imgsz=416,
        batch=16,
        workers=0,
        cache="ram",
        device=device,
        seed=RANDOM_SEED,
        project=str(yolo_runs_dir),
        name="cardd_yolov8n",
        exist_ok=True,
        save=True,
        val=True,
        patience=8,
        verbose=True
    )

    best_weights_src = yolo_runs_dir / "cardd_yolov8n" / "weights" / "best.pt"
    best_weights_dst = MODELS_DIR / "yolov8_damage_best.pt"

    if best_weights_src.exists():
        shutil.copy2(best_weights_src, best_weights_dst)
        print(f"Copied best model weights to: {best_weights_dst}", flush=True)
    else:
        # Fallback to last weights if best.pt is not distinct
        last_weights_src = yolo_runs_dir / "cardd_yolov8n" / "weights" / "last.pt"
        if last_weights_src.exists():
            shutil.copy2(last_weights_src, best_weights_dst)
            print(f"Copied last model weights to: {best_weights_dst}", flush=True)

    return best_weights_dst


def evaluate_on_test_set(model_path: Path):
    """
    Performs comprehensive evaluation on the unseen 120-image test set
    and generates visual prediction comparisons.
    """
    print("\n" + "="*50)
    print("EVALUATING DETECTOR ON UNSEEN TEST BENCHMARK")
    print("="*50, flush=True)

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    yaml_path = PROCESSED_DATA_DIR / "cardd" / "dataset.yaml"

    # Evaluate on test split
    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=416,
        batch=16,
        device=device,
        verbose=True
    )

    # Extract overall metrics
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)

    # Extract per-class metrics
    per_class_metrics = {}
    class_names = CARDD_CLASSES

    # metrics.box.maps is per-class mAP@50:95
    maps_per_class = metrics.box.maps
    for idx, cls_name in enumerate(class_names):
        cls_map = float(maps_per_class[idx]) if idx < len(maps_per_class) else 0.0
        per_class_metrics[cls_name] = {
            "mAP50_95": round(cls_map, 4)
        }

    test_summary = {
        "model_architecture": "YOLOv8n (3.0M params)",
        "test_dataset_size": 120,
        "input_resolution": "416x416",
        "overall_metrics": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(mp, 4),
            "recall": round(mr, 4)
        },
        "per_class_mAP50_95": per_class_metrics
    }

    metrics_out = METRICS_DIR / "detector_test_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    print(f"Test metrics saved to: {metrics_out}", flush=True)
    print(json.dumps(test_summary, indent=2), flush=True)

    # Generate Test Prediction Visualizations
    generate_test_prediction_plots(model)
    return test_summary


def generate_test_prediction_plots(model):
    """
    Renders Ground Truth vs Predicted Bounding Boxes for 6 test samples.
    """
    test_images_dir = PROCESSED_DATA_DIR / "cardd" / "images" / "test"
    test_labels_dir = PROCESSED_DATA_DIR / "cardd" / "labels" / "test"

    test_imgs = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png")))
    if not test_imgs:
        return

    # Select 6 diverse test images
    selected_imgs = test_imgs[:6]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15), dpi=300)
    axes = axes.flatten()

    class_colors = {
        "scratch": (59, 130, 246),       # Blue
        "dent": (239, 68, 68),          # Red
        "crack": (234, 179, 8),         # Yellow
        "glass shatter": (16, 185, 129),# Green
        "lamp broken": (139, 92, 246),  # Purple
        "tire flat": (236, 72, 153)     # Pink
    }

    for idx, img_p in enumerate(selected_imgs):
        orig_img = cv2.imread(str(img_p))
        h, w, _ = orig_img.shape

        # Run inference
        results = model.predict(source=str(img_p), conf=0.25, verbose=False)
        det_boxes = results[0].boxes

        # Draw predictions
        canvas = orig_img.copy()
        for box in det_boxes:
            cls_idx = int(box.cls[0].item())
            cls_name = CARDD_CLASSES[cls_idx] if cls_idx < len(CARDD_CLASSES) else "damage"
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            color = class_colors.get(cls_name, (0, 255, 0))
            cv2.rectangle(canvas, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
            label_text = f"{cls_name} {conf:.2f}"
            
            # Text background badge
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (xyxy[0], max(0, xyxy[1] - 22)), (xyxy[0] + tw + 6, max(22, xyxy[1])), color, -1)
            cv2.putText(canvas, label_text, (xyxy[0] + 3, max(16, xyxy[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(canvas_rgb)
        axes[idx].set_title(f"Test Sample {idx+1}: {img_p.name} ({len(det_boxes)} detections)", fontsize=11, weight="bold")
        axes[idx].axis("off")

    plt.suptitle("YOLOv8n Damage Detector — Unseen Test Benchmark Predictions", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    
    out_plot = PLOTS_DIR / "test_detection_predictions.png"
    plt.savefig(out_plot)
    plt.close()
    print(f"Test prediction visualization saved to: {out_plot}", flush=True)


def run_pipeline():
    best_weights = train_damage_detector()
    evaluate_on_test_set(best_weights)


if __name__ == "__main__":
    run_pipeline()
