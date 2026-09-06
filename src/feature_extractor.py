"""
Deep CNN Feature Embedding & Geometric Feature Extraction Pipeline.
Extracts 2048-dim ResNet50 visual embeddings and 10-dim domain geometric features.
"""

import os
import sys
import json
from pathlib import Path

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR, FEATURES_DIR, METRICS_DIR,
    CARDD_CLASSES, CARDD_CLASS_TO_IDX,
    SEVERITY_CLASSES, SEVERITY_CLASS_TO_IDX,
    NORM_MEAN, NORM_STD, RANDOM_SEED
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class DeepFeatureExtractor:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Initializing ResNet50 Feature Extractor on device: {self.device}", flush=True)
        
        # Load pretrained ResNet50 with DEFAULT (IMAGENET1K_V2) weights
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Identity()  # Output raw 2048-dim embedding vector
        self.model.eval()
        self.model.to(self.device)

        # Standard ImageNet preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])

    @torch.no_grad()
    def extract_image_embedding(self, image_path: Path) -> np.ndarray:
        """Extracts a 2048-dim L2-normalized feature embedding from a single image."""
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                feat = self.model(tensor)
                feat_np = feat.cpu().numpy().flatten()
                norm = np.linalg.norm(feat_np)
                if norm > 0:
                    feat_np = feat_np / norm
                return feat_np
        except Exception as e:
            return np.zeros(2048, dtype=np.float32)

    @torch.no_grad()
    def extract_batch(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        """Extracts 2048-dim L2-normalized embeddings in vectorized batches for speed."""
        all_feats = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            valid_indices = []
            for idx, p in enumerate(batch_paths):
                try:
                    with Image.open(p) as img:
                        img_rgb = img.convert("RGB")
                        t = self.transform(img_rgb)
                        tensors.append(t)
                        valid_indices.append(idx)
                except Exception:
                    pass

            if tensors:
                batch_tensor = torch.stack(tensors).to(self.device)
                batch_out = self.model(batch_tensor).cpu().numpy()
                # Row-wise L2 norm
                norms = np.linalg.norm(batch_out, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                batch_normed = batch_out / norms

                # Pad failed images with zero vector if any failed
                full_batch = np.zeros((len(batch_paths), 2048), dtype=np.float32)
                for valid_i, row in zip(valid_indices, batch_normed):
                    full_batch[valid_i] = row
                all_feats.append(full_batch)
            else:
                all_feats.append(np.zeros((len(batch_paths), 2048), dtype=np.float32))

        return np.vstack(all_feats) if all_feats else np.empty((0, 2048), dtype=np.float32)



def extract_geometric_features_from_yolo(label_path: Path) -> np.ndarray:
    """
    Computes a 10-dimensional structured geometric damage feature vector from YOLO label file:
    [box_count, total_area_ratio, max_area_ratio, mean_aspect_ratio,
     count_dent, count_scratch, count_crack, count_glass_shatter, count_lamp_broken, count_tire_flat]
    """
    features = np.zeros(10, dtype=np.float32)
    if not label_path.exists():
        return features

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return features

    areas = []
    aspect_ratios = []
    class_counts = [0] * len(CARDD_CLASSES)

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            w = float(parts[3])
            h = float(parts[4])
            area = w * h
            aspect = w / max(h, 1e-4)

            areas.append(area)
            aspect_ratios.append(aspect)
            if 0 <= cls_id < len(CARDD_CLASSES):
                class_counts[cls_id] += 1

    box_count = len(areas)
    total_area = sum(areas)
    max_area = max(areas) if areas else 0.0
    mean_aspect = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0

    features[0] = float(box_count)
    features[1] = float(total_area)
    features[2] = float(max_area)
    features[3] = float(mean_aspect)
    for i in range(len(CARDD_CLASSES)):
        features[4 + i] = float(class_counts[i])

    return features


def process_severity_features(extractor: DeepFeatureExtractor) -> dict:
    """
    Extracts deep visual embeddings for the Severity Benchmark (train, val, test splits).
    """
    print("\n" + "="*50)
    print("--- [Severity Benchmark] Extracting Deep CNN Embeddings ---")
    print("="*50, flush=True)

    sev_base = PROCESSED_DATA_DIR / "severity"
    split_summaries = {}

    for split in ["train", "val", "test"]:
        split_dir = sev_base / split
        img_paths = []
        labels_list = []
        filenames_list = []

        for cls_name in SEVERITY_CLASSES:
            cls_dir = split_dir / cls_name
            cls_idx = SEVERITY_CLASS_TO_IDX[cls_name]
            files = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg")))

            for img_p in files:
                img_paths.append(img_p)
                labels_list.append(cls_idx)
                filenames_list.append(img_p.name)

        print(f"Extracting {len(img_paths)} {split.upper()} severity embeddings in batches...", flush=True)
        X_emb = extractor.extract_batch(img_paths, batch_size=32)
        y = np.array(labels_list, dtype=np.int64)

        # Integrity assertion
        assert not np.isnan(X_emb).any(), f"NaN detected in {split} severity embeddings"
        assert not np.isinf(X_emb).any(), f"Inf detected in {split} severity embeddings"

        out_file = FEATURES_DIR / f"severity_features_{split}.npz"
        np.savez_compressed(
            out_file,
            features=X_emb,
            labels=y,
            filenames=filenames_list
        )

        split_summaries[split] = {
            "samples": len(X_emb),
            "feature_dim": X_emb.shape[1],
            "file": str(out_file)
        }
        print(f"Saved {split.upper()} Severity features: shape {X_emb.shape} to {out_file.name}", flush=True)

    return split_summaries


def process_cardd_features(extractor: DeepFeatureExtractor) -> dict:
    """
    Extracts deep visual embeddings and 10-dim geometric damage features for CarDD dataset.
    """
    print("\n" + "="*50)
    print("--- [CarDD Benchmark] Extracting Visual & Geometric Features ---")
    print("="*50, flush=True)

    cardd_base = PROCESSED_DATA_DIR / "cardd"
    split_summaries = {}

    for split in ["train", "val", "test"]:
        imgs_dir = cardd_base / "images" / split
        lbls_dir = cardd_base / "labels" / split

        img_files = sorted(list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png")))
        geom_features_list = []
        filenames_list = []

        for img_p in img_files:
            lbl_p = lbls_dir / f"{img_p.stem}.txt"
            feat_geom = extract_geometric_features_from_yolo(lbl_p)
            geom_features_list.append(feat_geom)
            filenames_list.append(img_p.name)

        print(f"Extracting {len(img_files)} {split.upper()} CarDD CNN embeddings in batches...", flush=True)
        X_cnn = extractor.extract_batch(img_files, batch_size=32)
        X_geom = np.array(geom_features_list, dtype=np.float32)

        # Integrity assertions
        assert not np.isnan(X_cnn).any(), f"NaN detected in {split} CarDD CNN embeddings"
        assert not np.isnan(X_geom).any(), f"NaN detected in {split} CarDD geometric features"

        out_file = FEATURES_DIR / f"cardd_features_{split}.npz"
        np.savez_compressed(
            out_file,
            cnn_features=X_cnn,
            geom_features=X_geom,
            filenames=filenames_list
        )

        split_summaries[split] = {
            "samples": len(X_cnn),
            "cnn_dim": X_cnn.shape[1],
            "geom_dim": X_geom.shape[1],
            "file": str(out_file)
        }
        print(f"Saved {split.upper()} CarDD features: CNN shape {X_cnn.shape}, Geom shape {X_geom.shape} to {out_file.name}", flush=True)

    return split_summaries


def run_feature_extraction():
    """Executes feature extraction across both benchmarks and saves summary metadata."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    extractor = DeepFeatureExtractor()

    severity_meta = process_severity_features(extractor)
    cardd_meta = process_cardd_features(extractor)

    summary = {
        "backbone_model": "ResNet50 (IMAGENET1K_V2, 2048-dim)",
        "l2_normalized": True,
        "geometric_feature_dim": 10,
        "geometric_feature_names": [
            "box_count", "total_area_ratio", "max_area_ratio", "mean_aspect_ratio",
            "count_dent", "count_scratch", "count_crack", "count_glass_shatter", "count_lamp_broken", "count_tire_flat"
        ],
        "severity_benchmark": severity_meta,
        "cardd_benchmark": cardd_meta
    }

    summary_file = METRICS_DIR / "phase5_feature_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print("PHASE 5 FEATURE EXTRACTION COMPLETE")
    print(f"Feature metadata saved to: {summary_file}")
    print("="*50, flush=True)
    return summary


if __name__ == "__main__":
    run_feature_extraction()
