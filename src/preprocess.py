"""
Preprocessing, format conversion, and zero-leakage dataset partitioning pipeline.
Prepares datasets for YOLOv8 Damage Detection and Multi-Class Severity Classification.
"""

import os
import sys
import json
import shutil
import hashlib
import random
from pathlib import Path
from collections import Counter

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import yaml
from PIL import Image

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, METRICS_DIR, RANDOM_SEED,
    CARDD_CLASSES, CARDD_CLASS_TO_IDX, CARDD_IDX_TO_CLASS,
    SEVERITY_CLASSES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


def compute_file_hash(filepath: Path) -> str:
    """Computes MD5 hash for image integrity and leakage checks."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_cardd_detection_dataset():
    """
    Converts CarDD FiftyOne annotations into standard YOLOv8 format,
    creates train/val/test splits (70/15/15), and generates dataset.yaml.
    """
    print("\n" + "="*50)
    print("--- [CarDD Detection] Preprocessing & Formatting ---")
    print("="*50, flush=True)

    cardd_raw_dir = RAW_DATA_DIR / "cardd"
    cardd_images_raw = cardd_raw_dir / "images"
    annotations_file = cardd_raw_dir / "annotations.json"

    out_dir = PROCESSED_DATA_DIR / "cardd"
    
    # Clean/create target structure
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    with open(annotations_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Filter only samples with valid image files and at least one valid bounding box
    valid_samples = []
    for s in samples:
        fname = Path(s.get("filepath", "")).name
        img_p = cardd_images_raw / fname
        if not img_p.exists():
            continue
        dets = s.get("detections", {}).get("detections", [])
        valid_dets = [d for d in dets if d.get("label", "").strip().lower() in CARDD_CLASS_TO_IDX]
        if valid_dets:
            valid_samples.append((img_p, valid_dets, s))

    print(f"Total valid CarDD detection samples: {len(valid_samples)}", flush=True)

    # Deterministic Shuffle
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(valid_samples)

    n_total = len(valid_samples)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    splits = {
        "train": valid_samples[:n_train],
        "val": valid_samples[n_train:n_train + n_val],
        "test": valid_samples[n_train + n_val:]
    }

    split_stats = {}
    split_hashes = {"train": set(), "val": set(), "test": set()}

    for split_name, items in splits.items():
        img_dest_dir = out_dir / "images" / split_name
        lbl_dest_dir = out_dir / "labels" / split_name

        class_counter = Counter()
        total_boxes = 0

        for img_path, dets, raw_sample in items:
            # File copy
            dest_img_path = img_dest_dir / img_path.name
            if not dest_img_path.exists():
                shutil.copy2(img_path, dest_img_path)

            # Record hash for leakage verification
            f_hash = compute_file_hash(dest_img_path)
            split_hashes[split_name].add(f_hash)

            # Generate YOLO TXT labels
            label_lines = []
            for d in dets:
                lbl = d.get("label", "").strip().lower()
                cls_id = CARDD_CLASS_TO_IDX[lbl]
                bx = d.get("bounding_box", [])
                if len(bx) == 4:
                    x_min, y_min, w, h = bx
                    # Clip coordinates to [0, 1]
                    x_min = max(0.0, min(1.0, float(x_min)))
                    y_min = max(0.0, min(1.0, float(y_min)))
                    w = max(0.001, min(1.0 - x_min, float(w)))
                    h = max(0.001, min(1.0 - y_min, float(h)))

                    # Convert (x_min, y_min, w, h) -> (x_center, y_center, w, h)
                    x_center = x_min + (w / 2.0)
                    y_center = y_min + (h / 2.0)

                    label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                    class_counter[lbl] += 1
                    total_boxes += 1

            lbl_path = lbl_dest_dir / f"{img_path.stem}.txt"
            with open(lbl_path, "w", encoding="utf-8") as f_lbl:
                f_lbl.write("\n".join(label_lines) + "\n")

        split_stats[split_name] = {
            "num_images": len(items),
            "num_boxes": total_boxes,
            "class_distribution": dict(class_counter)
        }
        print(f"[{split_name.upper()}] Images: {len(items)}, Boxes: {total_boxes}, Distribution: {dict(class_counter)}", flush=True)

    # Leakage check: verify disjoint hash sets
    train_val_leak = split_hashes["train"].intersection(split_hashes["val"])
    train_test_leak = split_hashes["train"].intersection(split_hashes["test"])
    val_test_leak = split_hashes["val"].intersection(split_hashes["test"])

    assert len(train_val_leak) == 0, f"CarDD Data Leakage Detected between Train & Val: {len(train_val_leak)} hashes"
    assert len(train_test_leak) == 0, f"CarDD Data Leakage Detected between Train & Test: {len(train_test_leak)} hashes"
    assert len(val_test_leak) == 0, f"CarDD Data Leakage Detected between Val & Test: {len(val_test_leak)} hashes"
    print(">>> [CarDD] Zero Data-Leakage Assertion: PASSED (Train, Val, Test are completely disjoint).", flush=True)

    # Generate dataset.yaml for Ultralytics YOLOv8
    dataset_yaml_content = {
        "path": str(out_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CARDD_CLASSES),
        "names": CARDD_CLASSES
    }

    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f_yaml:
        yaml.dump(dataset_yaml_content, f_yaml, default_flow_style=False)

    print(f"Saved YOLO dataset config to: {yaml_path}", flush=True)
    return split_stats, yaml_path


def process_severity_dataset():
    """
    Performs stratified 70/15/15 split on Severity benchmark images
    and organizes them into standard ImageFolder train/val/test directories.
    """
    print("\n" + "="*50)
    print("--- [Severity Classification] Preprocessing & Stratified Splitting ---")
    print("="*50, flush=True)

    sev_raw_dir = RAW_DATA_DIR / "severity"
    out_dir = PROCESSED_DATA_DIR / "severity"

    # Clean/create target structure
    for split in ["train", "val", "test"]:
        for cls_name in SEVERITY_CLASSES:
            (out_dir / split / cls_name).mkdir(parents=True, exist_ok=True)

    split_stats = {"train": {}, "val": {}, "test": {}}
    split_hashes = {"train": set(), "val": set(), "test": set()}

    for cls_name in SEVERITY_CLASSES:
        cls_raw_folder = sev_raw_dir / cls_name
        all_imgs = sorted(list(cls_raw_folder.glob("*.jpg")) + list(cls_raw_folder.glob("*.png")) + list(cls_raw_folder.glob("*.jpeg")))
        
        # Deterministic Shuffle per class
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(all_imgs)

        n_total = len(all_imgs)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val

        cls_splits = {
            "train": all_imgs[:n_train],
            "val": all_imgs[n_train:n_train + n_val],
            "test": all_imgs[n_train + n_val:]
        }

        for split_name, img_files in cls_splits.items():
            dest_dir = out_dir / split_name / cls_name
            for img_p in img_files:
                dest_p = dest_dir / img_p.name
                if not dest_p.exists():
                    shutil.copy2(img_p, dest_p)
                f_hash = compute_file_hash(dest_p)
                split_hashes[split_name].add(f_hash)

            split_stats[split_name][cls_name] = len(img_files)

    for split_name in ["train", "val", "test"]:
        total_in_split = sum(split_stats[split_name].values())
        print(f"[{split_name.upper()}] Total: {total_in_split}, Breakdown: {split_stats[split_name]}", flush=True)

    # Leakage check: verify disjoint hash sets
    train_val_leak = split_hashes["train"].intersection(split_hashes["val"])
    train_test_leak = split_hashes["train"].intersection(split_hashes["test"])
    val_test_leak = split_hashes["val"].intersection(split_hashes["test"])

    assert len(train_val_leak) == 0, f"Severity Data Leakage Detected between Train & Val: {len(train_val_leak)} hashes"
    assert len(train_test_leak) == 0, f"Severity Data Leakage Detected between Train & Test: {len(train_test_leak)} hashes"
    assert len(val_test_leak) == 0, f"Severity Data Leakage Detected between Val & Test: {len(val_test_leak)} hashes"
    print(">>> [Severity] Zero Data-Leakage Assertion: PASSED (Train, Val, Test are completely disjoint).", flush=True)

    return split_stats


def run_preprocessing_pipeline():
    """Executes end-to-end preprocessing, verification, and manifest generation."""
    cardd_stats, yaml_path = process_cardd_detection_dataset()
    severity_stats = process_severity_dataset()

    manifest = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },
        "cardd_detection_splits": cardd_stats,
        "cardd_yaml_config": str(yaml_path),
        "severity_classification_splits": severity_stats,
        "zero_leakage_verified": True
    }

    manifest_path = METRICS_DIR / "phase3_split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "="*50)
    print("PREPROCESSING & ZERO-LEAKAGE SPLITTING COMPLETE")
    print(f"Manifest saved to: {manifest_path}")
    print("="*50, flush=True)
    return manifest


if __name__ == "__main__":
    run_preprocessing_pipeline()
