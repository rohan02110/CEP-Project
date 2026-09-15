"""
Training and Evaluation Pipeline for 100% Classical Damage Detector.
Constructs training samples via Selective Search + Ground Truth IoU matching,
extracts HOG+LBP+HSV handcrafted feature vectors, and trains a Random Forest Classifier.
Reports cross-validated per-class Precision, Recall, F1-Score, and Confusion Matrix.

Strict Academic Constraint: Zero deep learning dependencies (No CNNs, PyTorch, or TensorFlow).
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import (
    CARDD_CLASSES, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR, RANDOM_SEED
)
from src.classical_detector import (
    extract_patch_features, generate_selective_search_proposals,
    compute_iou, DETECTOR_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
)

np.random.seed(RANDOM_SEED)


def load_cardd_annotations(split: str = "train") -> List[Dict]:
    """
    Loads images and parsed ground-truth bounding boxes for a given CarDD split.
    """
    img_dir = PROCESSED_DATA_DIR / "cardd" / "images" / split
    lbl_dir = PROCESSED_DATA_DIR / "cardd" / "labels" / split

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"CarDD {split} split directory not found at {img_dir}")

    records = []
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    for img_path in img_files:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = []

        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_idx = int(parts[0])
                        xc, yc, w, h = [float(p) for p in parts[1:5]]
                        cls_name = CARDD_CLASSES[cls_idx] if cls_idx < len(CARDD_CLASSES) else "dent"
                        gt_boxes.append({
                            "class_name": cls_name,
                            "class_idx": cls_idx,
                            "yolo_box": (xc, yc, w, h)
                        })

        records.append({
            "image_path": img_path,
            "gt_boxes": gt_boxes
        })

    print(f"[DATA] Loaded {len(records)} images for '{split}' split ({sum(len(r['gt_boxes']) for r in records)} GT damage annotations).", flush=True)
    return records


def yolo_to_pixel_box(yolo_box: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Converts normalized [x_center, y_center, w, h] to pixel coordinates [x1, y1, x2, y2]."""
    xc, yc, w, h = yolo_box
    x1 = int(max(0, (xc - w / 2.0) * img_w))
    y1 = int(max(0, (yc - h / 2.0) * img_h))
    x2 = int(min(img_w, (xc + w / 2.0) * img_w))
    y2 = int(min(img_h, (yc + h / 2.0) * img_h))
    return (x1, y1, x2, y2)


def build_proposal_training_dataset(
    records: List[Dict],
    max_images: int = 400,
    max_proposals_per_img: int = 150,
    neg_to_pos_ratio: float = 2.5
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generates training patches and labels:
    - Extracts ground-truth bounding box crops as positive samples.
    - Generates Selective Search proposals.
    - Matches proposals with GT boxes via IoU:
      * IoU >= 0.50 -> Positive instance of matching class.
      * IoU < 0.20 -> Negative (Background) instance.
      * 0.20 <= IoU < 0.50 -> Discarded (ambiguous overlap).
    - Subsamples background to maintain balanced class representation.
    """
    print(f"\n[DATASET BUILD] Generating Selective Search proposals & extracting 376-dim handcrafted features...", flush=True)
    start_time = time.time()

    features_list = []
    labels_list = []

    pos_count = 0
    bg_count = 0
    class_counts = {c: 0 for c in DETECTOR_CLASSES}

    selected_records = records[:max_images] if max_images else records

    for idx, rec in enumerate(selected_records):
        img_path = rec["image_path"]
        cv_img_bgr = cv2.imread(str(img_path))
        if cv_img_bgr is None:
            continue

        img_h, img_w = cv_img_bgr.shape[:2]
        gt_pixel_boxes = []

        # 1. Add Exact Ground-Truth Positive Crops
        for gt in rec["gt_boxes"]:
            box_px = yolo_to_pixel_box(gt["yolo_box"], img_w, img_h)
            x1, y1, x2, y2 = box_px
            if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                crop = cv_img_bgr[y1:y2, x1:x2]
                feats = extract_patch_features(crop)
                features_list.append(feats)
                labels_list.append(gt["class_name"])
                class_counts[gt["class_name"]] += 1
                pos_count += 1
                gt_pixel_boxes.append((box_px, gt["class_name"]))

        # 2. Run Selective Search Region Proposals
        proposals = generate_selective_search_proposals(cv_img_bgr, max_proposals=max_proposals_per_img)

        img_pos = []
        img_neg = []

        for prop_box in proposals:
            x1, y1, x2, y2 = prop_box
            crop = cv_img_bgr[y1:y2, x1:x2]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue

            if not gt_pixel_boxes:
                # No GT boxes -> pure background
                img_neg.append((prop_box, "background"))
                continue

            # Compute IoU with all GT boxes on this image
            best_iou = 0.0
            best_gt_cls = None

            for gt_box, gt_cls in gt_pixel_boxes:
                iou = compute_iou(prop_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_cls = gt_cls

            if best_iou >= 0.50 and best_gt_cls is not None:
                img_pos.append((prop_box, best_gt_cls))
            elif best_iou < 0.20:
                img_neg.append((prop_box, "background"))
            # Ambiguous (0.20 <= IoU < 0.50) is discarded

        # Add positive proposal crops
        for prop_box, cls_name in img_pos:
            x1, y1, x2, y2 = prop_box
            crop = cv_img_bgr[y1:y2, x1:x2]
            feats = extract_patch_features(crop)
            features_list.append(feats)
            labels_list.append(cls_name)
            class_counts[cls_name] += 1
            pos_count += 1

        # Subsample negative background crops per image
        max_neg_for_img = max(3, int(max(1, len(img_pos) + len(gt_pixel_boxes)) * neg_to_pos_ratio))
        if len(img_neg) > max_neg_for_img:
            indices = np.random.choice(len(img_neg), size=max_neg_for_img, replace=False)
            selected_neg = [img_neg[i] for i in indices]
        else:
            selected_neg = img_neg

        for prop_box, cls_name in selected_neg:
            x1, y1, x2, y2 = prop_box
            crop = cv_img_bgr[y1:y2, x1:x2]
            feats = extract_patch_features(crop)
            features_list.append(feats)
            labels_list.append(cls_name)
            class_counts[cls_name] += 1
            bg_count += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(selected_records):
            print(f"  [PROPOSALS] Processed {idx + 1}/{len(selected_records)} images -> {len(features_list)} samples extracted.", flush=True)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list)

    elapsed = time.time() - start_time
    print(f"[DATASET BUILD COMPLETE] Extracted {X.shape[0]} total samples across {X.shape[1]} features in {elapsed:.1f}s.")
    print("Class distribution:")
    for cls_name, cnt in class_counts.items():
        print(f"  • {cls_name:<16}: {cnt:>5} samples ({cnt/len(y)*100:.1f}%)")

    return X, y, DETECTOR_CLASSES


def train_and_benchmark_classical_detector():
    """
    Main routine:
    1. Loads dataset and extracts handcrafted features.
    2. Runs 5-Fold Stratified Cross-Validation.
    3. Trains final Random Forest model and serializes artifact.
    4. Plots per-class confusion matrix and saves JSON benchmark metrics.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAINING 100% CLASSICAL COMPUTER VISION & RANDOM FOREST DAMAGE DETECTOR")
    print("=" * 80, flush=True)

    # 1. Load CarDD dataset splits
    train_records = load_cardd_annotations("train")
    val_records = load_cardd_annotations("val")

    # Combine train + val for robust cross-validated training
    all_records = train_records + val_records

    # 2. Build training samples via Selective Search + GT extraction
    X, y, classes = build_proposal_training_dataset(
        all_records,
        max_images=len(all_records),
        max_proposals_per_img=150,
        neg_to_pos_ratio=2.5
    )

    # 3. 5-Fold Stratified Cross Validation
    print("\n" + "-" * 80)
    print("EXECUTING 5-FOLD STRATIFIED CROSS-VALIDATION ON HANDCRAFTED FEATURES")
    print("-" * 80, flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.empty_like(y)
    oof_probs = np.zeros((len(y), len(classes)), dtype=np.float32)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        rf_fold = RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            class_weight='balanced_subsample',
            random_state=RANDOM_SEED + fold,
            n_jobs=-1
        )
        rf_fold.fit(X_tr, y_tr)

        val_preds = rf_fold.predict(X_va)
        val_probs = rf_fold.predict_proba(X_va)

        oof_preds[val_idx] = val_preds
        
        # Align class probabilities
        model_classes = list(rf_fold.classes_)
        for i, c in enumerate(model_classes):
            global_idx = classes.index(c)
            oof_probs[val_idx, global_idx] = val_probs[:, i]

        p, r, f1, _ = precision_recall_fscore_support(y_va, val_preds, average='macro', zero_division=0)
        print(f"  Fold {fold + 1}/5 -> Macro Precision: {p:.4f} | Macro Recall: {r:.4f} | Macro F1: {f1:.4f}", flush=True)
        fold_metrics.append({"fold": fold + 1, "macro_precision": p, "macro_recall": r, "macro_f1": f1})

    # Overall OOF Classification Report
    print("\n" + "=" * 80)
    print("OUT-OF-FOLD (OOF) CROSS-VALIDATED CLASSIFICATION REPORT")
    print("=" * 80)
    report_dict = classification_report(y, oof_preds, target_names=classes, output_dict=True, zero_division=0)
    report_str = classification_report(y, oof_preds, target_names=classes, zero_division=0)
    print(report_str)

    # 4. Train Final Production Random Forest Model on Full Dataset
    print("\n[TRAINING FINAL MODEL] Fitting Random Forest on all extracted training samples...")
    final_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_split=4,
        class_weight='balanced_subsample',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    final_rf.fit(X, y)

    # 5. Serialize Champion Classical Detector Checkpoint
    checkpoint_path = MODELS_DIR / "classical_damage_rf_best.joblib"
    joblib.dump({
        "model_name": "Classical CV + Random Forest Damage Detector",
        "model": final_rf,
        "class_names": list(final_rf.classes_),
        "feature_dim": X.shape[1],
        "feature_schema": {
            "hog_dims": 324,
            "lbp_dims": 20,
            "hsv_dims": 32,
            "total_dims": 376
        },
        "oof_classification_report": report_dict,
        "sample_count": len(y)
    }, checkpoint_path)
    print(f"[CHECKPOINT] Saved Classical Detector Artifact to: {checkpoint_path}")

    # 6. Plot & Save Confusion Matrix
    cm = confusion_matrix(y, oof_preds, labels=classes)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar_kws={'label': 'Normalized Rate'}
    )
    ax.set_title("Classical CV + Random Forest Detector: 5-Fold CV Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Ground-Truth Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = PLOTS_DIR / "classical_detector_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved Confusion Matrix to: {cm_path}")

    # 7. Serialize Benchmark Metrics to JSON
    benchmark_json = {
        "model_architecture": "Selective Search Fast + HOG(324) + LBP(20) + HSV(32) + Random Forest(150 trees)",
        "deep_learning_dependencies": False,
        "feature_vector_dimension": 376,
        "total_training_crops": len(y),
        "class_distribution": {c: int(np.sum(y == c)) for c in classes},
        "cross_validation": {
            "n_folds": 5,
            "macro_precision": float(np.mean([f["macro_precision"] for f in fold_metrics])),
            "macro_recall": float(np.mean([f["macro_recall"] for f in fold_metrics])),
            "macro_f1": float(np.mean([f["macro_f1"] for f in fold_metrics]))
        },
        "per_class_metrics": {
            cls_name: {
                "precision": float(report_dict[cls_name]["precision"]),
                "recall": float(report_dict[cls_name]["recall"]),
                "f1_score": float(report_dict[cls_name]["f1-score"]),
                "support": int(report_dict[cls_name]["support"])
            }
            for cls_name in classes if cls_name in report_dict
        }
    }

    metrics_file = METRICS_DIR / "classical_detector_benchmark.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_json, f, indent=2)
    print(f"[METRICS] Saved Full Benchmark JSON to: {metrics_file}")

    print("\n" + "=" * 80)
    print("[SUCCESS] CLASSICAL DAMAGE DETECTOR TRAINING & BENCHMARK COMPLETE!")
    print("=" * 80)
    return benchmark_json


if __name__ == "__main__":
    train_and_benchmark_classical_detector()
