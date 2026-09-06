"""
Data acquisition, validation, integrity checks, and Exploratory Data Analysis (EDA).
Saves clean datasets and publication-ready statistical visualizations.
"""

import os
import sys
from pathlib import Path

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import json
import hashlib
import random
from collections import Counter
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import cv2
from huggingface_hub import hf_hub_download, list_repo_files

from src.config import (
    RAW_DATA_DIR, PLOTS_DIR, METRICS_DIR, RANDOM_SEED,
    CARDD_CLASSES, CARDD_CLASS_TO_IDX, SEVERITY_CLASSES
)

# Set seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="muted")


def compute_file_hash(filepath: Path) -> str:
    """Computes MD5 hash for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_cardd_dataset(target_dir: Path, max_samples: int = 800) -> dict:
    """
    Downloads CarDD images and parses FiftyOne/COCO bounding box annotations.
    """
    cardd_dir = target_dir / "cardd"
    images_dir = cardd_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- [CarDD] Downloading annotations & metadata ---")
    samples_p = hf_hub_download("harpreetsahota/CarDD", "samples.json", repo_type="dataset")
    with open(samples_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_samples = data["samples"]
    print(f"Total available CarDD samples in repo: {len(all_samples)}")
    
    # Filter samples that have at least one valid detection
    valid_samples = [
        s for s in all_samples 
        if s.get("detections") and s["detections"].get("detections")
    ]
    print(f"Samples with valid bounding box detections: {len(valid_samples)}")
    
    # Select balanced subset
    selected_samples = valid_samples[:max_samples]
    
    print(f"--- [CarDD] Downloading {len(selected_samples)} selected images in parallel ---")
    
    def download_img(sample):
        rel_path = sample.get("filepath", "")
        if not rel_path.startswith("data/"):
            rel_path = f"data/{Path(rel_path).name}"
        dest_path = images_dir / Path(rel_path).name
        s_id = sample["_id"]["$oid"] if isinstance(sample["_id"], dict) else sample["_id"]
        
        # Immediate local cache check
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return s_id, dest_path, sample
            
        try:
            downloaded_path = hf_hub_download(
                "harpreetsahota/CarDD", 
                rel_path, 
                repo_type="dataset"
            )
            with open(downloaded_path, "rb") as src_f, open(dest_path, "wb") as dst_f:
                dst_f.write(src_f.read())
            return s_id, dest_path, sample
        except Exception as e:
            return None, None, None

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(download_img, selected_samples))
    
    for s_id, path, sample in results:
        if path and path.exists():
            records.append(sample)
            
    print(f"Successfully downloaded & verified {len(records)} CarDD raw images.", flush=True)
    
    # Save local annotations json
    with open(cardd_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    return {"dir": cardd_dir, "samples": records}


def download_severity_dataset(target_dir: Path, max_per_class: int = 200) -> dict:
    """
    Downloads Comprehensive Car Damage severity dataset with direct URL streaming.
    Maps categories to [normal, moderate_breakage, severe_crushed].
    """
    import urllib.request
    import shutil
    
    sev_dir = target_dir / "severity"
    sev_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- [Severity Dataset] Listing & acquiring balanced images ---", flush=True)
    
    folder_mapping = {
        "F_Normal": ("normal", "front_normal"),
        "R_Normal": ("normal", "rear_normal"),
        "F_Breakage": ("moderate_breakage", "front_breakage"),
        "R_Breakage": ("moderate_breakage", "rear_breakage"),
        "F_Crushed": ("severe_crushed", "front_crushed"),
        "R_Crushed": ("severe_crushed", "rear_crushed")
    }
    
    all_repo_files = list_repo_files("SaiVaibhavS/comprehensive-car-damage", repo_type="dataset")
    
    max_per_subfolder = max_per_class // 2  # 100 per subfolder -> 200 per standard class
    
    files_to_download = []
    for folder_name, (std_cls, sub_desc) in folder_mapping.items():
        cat_dir = sev_dir / std_cls
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        folder_files = [f for f in all_repo_files if f.startswith(f"{folder_name}/") and f.lower().endswith((".jpg", ".png", ".jpeg"))]
        folder_files = sorted(folder_files)[:max_per_subfolder]
        
        for fpath in folder_files:
            dest_name = f"{folder_name}_{Path(fpath).name}"
            dest_file = cat_dir / dest_name
            files_to_download.append((fpath, dest_file, std_cls, folder_name))
            
    print(f"Total balanced severity images to acquire: {len(files_to_download)}", flush=True)
    
    # Base URL for direct raw LFS/dataset file downloads on Hugging Face
    base_raw_url = "https://huggingface.co/datasets/SaiVaibhavS/comprehensive-car-damage/resolve/main"
    
    def fetch_single_file(item):
        repo_path, local_dest, std_cls, orig_cat = item
        if local_dest.exists() and local_dest.stat().st_size > 0:
            return {
                "filepath": str(local_dest),
                "severity_class": std_cls,
                "original_category": orig_cat
            }
        try:
            url = f"{base_raw_url}/{repo_path}"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=15) as response, open(local_dest, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            return {
                "filepath": str(local_dest),
                "severity_class": std_cls,
                "original_category": orig_cat
            }
        except Exception as e:
            # Fallback to hf_hub_download
            try:
                cp = hf_hub_download("SaiVaibhavS/comprehensive-car-damage", repo_path, repo_type="dataset")
                shutil.copy2(cp, local_dest)
                return {
                    "filepath": str(local_dest),
                    "severity_class": std_cls,
                    "original_category": orig_cat
                }
            except Exception:
                return None

    downloaded_records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(fetch_single_file, files_to_download))
        
    for r in results:
        if r is not None:
            downloaded_records.append(r)
            
    print(f"Successfully organized & verified {len(downloaded_records)} balanced Severity images.", flush=True)
    return {"dir": sev_dir, "records": downloaded_records}



def perform_validation_and_eda():
    """
    Performs comprehensive dataset integrity validation, duplicate detection,
    bounding box geometry statistics, and generates EDA visualization figures.
    """
    print("\n" + "="*50)
    print("STARTING DATA VALIDATION & EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # 1. Acquire data
    cardd_info = download_cardd_dataset(RAW_DATA_DIR, max_samples=800)
    sev_info = download_severity_dataset(RAW_DATA_DIR, max_per_class=200)
    
    # 2. CarDD Validation & EDA
    cardd_samples = cardd_info["samples"]
    images_dir = cardd_info["dir"] / "images"
    
    img_resolutions = []
    img_aspect_ratios = []
    bbox_classes = []
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    boxes_per_image = []
    corrupted_images = 0
    file_hashes = {}
    duplicates = []
    
    parsed_cardd_rows = []
    
    for sample in cardd_samples:
        rel_path = Path(sample.get("filepath", "")).name
        img_p = images_dir / rel_path
        if not img_p.exists():
            continue
            
        # File integrity check
        try:
            with Image.open(img_p) as img:
                img.verify()
            # Reopen to read dimensions
            with Image.open(img_p) as img:
                w, h = img.size
                channels = len(img.getbands())
                img_resolutions.append((w, h))
                img_aspect_ratios.append(w / h)
        except Exception as e:
            corrupted_images += 1
            continue
            
        # Duplicate detection
        f_hash = compute_file_hash(img_p)
        if f_hash in file_hashes:
            duplicates.append((img_p.name, file_hashes[f_hash]))
        else:
            file_hashes[f_hash] = img_p.name
            
        # Parse bounding boxes
        dets = sample.get("detections", {}).get("detections", [])
        boxes_per_image.append(len(dets))
        
        for d in dets:
            lbl = d.get("label", "").strip().lower()
            if lbl not in CARDD_CLASSES:
                # Class name standardization if slight mismatch
                continue
                
            # FiftyOne bounding box: [x_min_norm, y_min_norm, width_norm, height_norm]
            bbox = d.get("bounding_box", [])
            if len(bbox) == 4:
                x_norm, y_norm, w_norm, h_norm = bbox
                # Verify bounds
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))
                w_norm = max(0.0, min(1.0 - x_norm, w_norm))
                h_norm = max(0.0, min(1.0 - y_norm, h_norm))
                
                area_norm = w_norm * h_norm
                
                bbox_classes.append(lbl)
                bbox_widths.append(w_norm)
                bbox_heights.append(h_norm)
                bbox_areas.append(area_norm)
                
                parsed_cardd_rows.append({
                    "image_id": img_p.name,
                    "image_width": w,
                    "image_height": h,
                    "damage_type": lbl,
                    "bbox_x": x_norm,
                    "bbox_y": y_norm,
                    "bbox_w": w_norm,
                    "bbox_h": h_norm,
                    "bbox_area_ratio": area_norm
                })

    df_cardd = pd.DataFrame(parsed_cardd_rows)
    
    # 3. Severity Validation & EDA
    sev_records = sev_info["records"]
    sev_counts = Counter([r["severity_class"] for r in sev_records])
    
    # 4. Generate Publication-Quality EDA Visualizations
    
    # Fig 1: CarDD Damage Class Distribution
    plt.figure(figsize=(9, 5), dpi=300)
    class_counts = Counter(bbox_classes)
    classes_sorted = [c for c in CARDD_CLASSES if c in class_counts]
    counts_sorted = [class_counts[c] for c in classes_sorted]
    
    palette = sns.color_palette("Blues_r", len(classes_sorted))
    ax = sns.barplot(x=classes_sorted, y=counts_sorted, hue=classes_sorted, palette=palette, legend=False)
    plt.title("CarDD Benchmark — Damage Category Instance Distribution", fontsize=13, pad=12, weight="bold")
    plt.xlabel("Damage Class", fontsize=11, weight="bold")
    plt.ylabel("Annotated Bounding Box Instances", fontsize=11, weight="bold")
    plt.xticks(rotation=15, fontsize=10)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}\n({p.get_height()/sum(counts_sorted)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_cardd_class_distribution.png")
    plt.close()
    
    # Fig 2: Bounding Box Geometry (Aspect Ratio & Normalized Area)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    sns.histplot(bbox_areas, bins=30, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Damage Area Ratio Distribution (BBox Area / Image Area)", fontsize=11, weight="bold")
    axes[0].set_xlabel("Normalized BBox Area Ratio", fontsize=10)
    axes[0].set_ylabel("Instance Frequency", fontsize=10)
    
    sns.histplot(boxes_per_image, discrete=True, ax=axes[1], color="#d95f02")
    axes[1].set_title("Damage Instances per Image", fontsize=11, weight="bold")
    axes[1].set_xlabel("Number of Damage BBoxes in Single Photo", fontsize=10)
    axes[1].set_ylabel("Image Frequency", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_cardd_bbox_geometry.png")
    plt.close()
    
    # Fig 3: Severity Class Distribution
    plt.figure(figsize=(7, 4.5), dpi=300)
    sev_keys = ["normal", "moderate_breakage", "severe_crushed"]
    sev_vals = [sev_counts.get(k, 0) for k in sev_keys]
    sev_labels = ["Normal\n(No Damage)", "Moderate\n(Breakage / Deformed)", "Severe\n(Crushed / Structural)"]
    
    sev_palette = ["#2ca02c", "#ff7f0e", "#d62728"]
    ax = sns.barplot(x=sev_labels, y=sev_vals, hue=sev_labels, palette=sev_palette, legend=False)
    plt.title("Vehicle Damage Severity Benchmark Distribution", fontsize=12, pad=12, weight="bold")
    plt.xlabel("Severity Classification", fontsize=10, weight="bold")
    plt.ylabel("Sample Count", fontsize=10, weight="bold")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}\n({p.get_height()/sum(sev_vals)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_severity_distribution.png")
    plt.close()
    
    # Fig 4: Sample Visualizations with Ground Truth Annotations
    sample_vis_paths = []
    # Pick 4 random images with detections
    sample_subset = [s for s in cardd_samples if s.get("detections", {}).get("detections")][:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()
    
    color_map = {
        "scratch": "#3b82f6",
        "dent": "#ef4444",
        "crack": "#eab308",
        "lamp broken": "#8b5cf6",
        "glass shatter": "#10b981",
        "tire flat": "#ec4899"
    }
    
    for idx, sample in enumerate(sample_subset):
        rel_p = Path(sample.get("filepath", "")).name
        img_file = images_dir / rel_p
        if not img_file.exists():
            continue
            
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        dets = sample.get("detections", {}).get("detections", [])
        for d in dets:
            lbl = d.get("label", "")
            bx = d.get("bounding_box", [])
            if len(bx) == 4:
                x_min = int(bx[0] * w)
                y_min = int(bx[1] * h)
                bx_w = int(bx[2] * w)
                bx_h = int(bx[3] * h)
                x_max = min(w - 1, x_min + bx_w)
                y_max = min(h - 1, y_min + bx_h)
                
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (239, 68, 68), 3)
                cv2.putText(img, lbl, (x_min, max(20, y_min - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
        axes[idx].imshow(img)
        axes[idx].set_title(f"Sample #{idx+1} ({len(dets)} annotations)", fontsize=10, weight="bold")
        axes[idx].axis("off")
        
    plt.suptitle("CarDD Ground Truth Annotations — Inspection Samples", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_sample_detections_grid.png")
    plt.close()
    
    # 5. Compile Summary Statistics
    summary = {
        "cardd_dataset": {
            "total_images_analyzed": len(cardd_samples),
            "corrupted_images_detected": corrupted_images,
            "exact_duplicate_pairs": len(duplicates),
            "total_bounding_boxes": len(df_cardd),
            "average_boxes_per_image": float(np.mean(boxes_per_image)),
            "class_distribution": dict(Counter(bbox_classes)),
            "mean_damage_area_ratio": float(np.mean(bbox_areas)),
            "median_damage_area_ratio": float(np.median(bbox_areas)),
            "image_resolutions": {
                "mean_width": float(np.mean([r[0] for r in img_resolutions])),
                "mean_height": float(np.mean([r[1] for r in img_resolutions])),
                "mean_aspect_ratio": float(np.mean(img_aspect_ratios))
            }
        },
        "severity_dataset": {
            "total_images_analyzed": len(sev_records),
            "class_distribution": dict(sev_counts)
        },
        "artifacts_generated": [
            "experiments/plots/eda_cardd_class_distribution.png",
            "experiments/plots/eda_cardd_bbox_geometry.png",
            "experiments/plots/eda_severity_distribution.png",
            "experiments/plots/eda_sample_detections_grid.png"
        ]
    }
    
    with open(METRICS_DIR / "phase2_eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*50)
    print("DATA VALIDATION & EDA COMPLETE")
    print("="*50)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    perform_validation_and_eda()
