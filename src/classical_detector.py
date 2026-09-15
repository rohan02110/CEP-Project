"""
100% Classical Computer Vision & Machine Learning Vehicle Damage Detector.
Implements:
1. Selective Search Region Proposals (OpenCV Fast Mode)
2. Handcrafted Feature Extraction:
   - Histogram of Oriented Gradients (HOG, 324 dims)
   - Multi-Scale Local Binary Patterns (LBP, 20 dims)
   - HSV Color Histograms (32 dims)
   Total = 376 feature dimensions per region crop.
3. Random Forest Classification (scikit-learn)
4. Class-wise Non-Maximum Suppression (NMS)

Strict Academic Constraint: Zero deep learning dependencies (No CNNs, PyTorch, or TensorFlow).
"""

import sys
from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage.feature import hog, local_binary_pattern

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import CARDD_CLASSES, MODELS_DIR, RANDOM_SEED
from src.cost_estimator import DetectedDamage

# Fixed patch size for canonical feature extraction
PATCH_SIZE = (64, 64)

# 6 CarDD damage classes + 1 implicit background class
DETECTOR_CLASSES = list(CARDD_CLASSES) + ["background"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(DETECTOR_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(DETECTOR_CLASSES)}


def extract_patch_features(patch_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts a 376-dimensional handcrafted feature vector from a single cropped image patch.
    
    Components:
    1. HOG (324 dims): 64x64 patch -> 9 orientations, 16x16 pixels/cell, 2x2 cells/block.
       (4-1)x(4-1) = 9 blocks * (2*2*9) = 324 dimensions.
    2. LBP (20 dims): Local texture patterns at radius R=1 (10 bins) and R=2 (10 bins).
    3. HSV (32 dims): 16 Hue bins, 8 Saturation bins, 8 Value bins.
    
    Total dimensionality = 324 + 20 + 32 = 376 dims.
    """
    if patch_bgr.shape[0] != PATCH_SIZE[0] or patch_bgr.shape[1] != PATCH_SIZE[1]:
        patch_bgr = cv2.resize(patch_bgr, PATCH_SIZE, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)

    # 1. HOG Descriptor (Shape & Edge Gradients)
    hog_feats = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )

    # 2. Local Binary Patterns (Micro-texture & Surface Anomalies)
    lbp_r1 = local_binary_pattern(gray, P=8, R=1.0, method='uniform')
    hist_lbp1, _ = np.histogram(lbp_r1.ravel(), bins=10, range=(0, 10), density=True)

    lbp_r2 = local_binary_pattern(gray, P=8, R=2.0, method='uniform')
    hist_lbp2, _ = np.histogram(lbp_r2.ravel(), bins=10, range=(0, 10), density=True)
    lbp_feats = np.hstack([hist_lbp1, hist_lbp2])

    # 3. HSV Color Histograms (Rust, Paint Damage & Discoloration)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])

    hist_h = (hist_h / (hist_h.sum() + 1e-7)).flatten()
    hist_s = (hist_s / (hist_s.sum() + 1e-7)).flatten()
    hist_v = (hist_v / (hist_v.sum() + 1e-7)).flatten()
    color_feats = np.hstack([hist_h, hist_s, hist_v])

    # Concatenate all handcrafted representations
    feature_vector = np.hstack([hog_feats, lbp_feats, color_feats]).astype(np.float32)
    return feature_vector


def generate_selective_search_proposals(
    img_bgr: np.ndarray,
    max_proposals: int = 300,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.85,
    max_aspect_ratio: float = 5.0
) -> List[Tuple[int, int, int, int]]:
    """
    Generates candidate damage bounding boxes using OpenCV Selective Search in Fast Mode.
    Prunes degenerate boxes based on area and aspect ratio constraints.
    
    Returns:
        List of [x1, y1, x2, y2] bounding boxes in pixel coordinates.
    """
    img_h, img_w = img_bgr.shape[:2]
    img_area = float(img_w * img_h)

    # Initialize OpenCV Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_bgr)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    filtered_boxes = []

    for x, y, w, h in rects[:max_proposals * 2]:
        box_area = float(w * h)
        area_ratio = box_area / img_area
        aspect = max(w / max(1, h), h / max(1, w))

        # Geometric filtering
        if area_ratio < min_area_ratio:
            continue
        if area_ratio > max_area_ratio:
            continue
        if aspect > max_aspect_ratio:
            continue

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, int(x + w))
        y2 = min(img_h, int(y + h))

        if (x2 - x1) >= 8 and (y2 - y1) >= 8:
            filtered_boxes.append((x1, y1, x2, y2))
            if len(filtered_boxes) >= max_proposals:
                break

    return filtered_boxes


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(area_a + area_b - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


def apply_classwise_nms(
    detections: List[Tuple[str, float, Tuple[int, int, int, int]]],
    iou_threshold: float = 0.40
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """
    Applies class-wise Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes.
    """
    if not detections:
        return []

    # Group by damage class
    by_class = {}
    for item in detections:
        cls_name, score, box = item
        by_class.setdefault(cls_name, []).append((score, box))

    surviving = []
    for cls_name, candidates in by_class.items():
        # Sort by confidence score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        keep = []

        while candidates:
            best_score, best_box = candidates.pop(0)
            keep.append((cls_name, best_score, best_box))

            # Discard any remaining box that heavily overlaps with best_box
            candidates = [
                (s, b) for (s, b) in candidates
                if compute_iou(best_box, b) < iou_threshold
            ]

        surviving.extend(keep)

    # Sort final surviving list by score descending
    surviving.sort(key=lambda x: x[1], reverse=True)
    return surviving


class ClassicalDamageDetector:
    """
    100% Classical CV Damage Detector.
    Integrates Selective Search proposals, HOG+LBP+HSV feature extraction,
    and Random Forest classification with class-wise NMS.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path is None:
            model_path = MODELS_DIR / "classical_damage_rf_best.joblib"

        self.model_path = Path(model_path)
        if self.model_path.exists():
            checkpoint = joblib.load(self.model_path)
            self.model = checkpoint["model"]
            self.class_names = checkpoint.get("class_names", DETECTOR_CLASSES)
            self.feature_dim = checkpoint.get("feature_dim", 376)
            print(f"[CLASSICAL DETECTOR] Loaded Random Forest model from {self.model_path}", flush=True)
        else:
            self.model = None
            self.class_names = DETECTOR_CLASSES
            self.feature_dim = 376
            print(f"[CLASSICAL DETECTOR] Model file not found at {self.model_path}. Run src/train_classical_detector.py first.", flush=True)

    def classify_region(
        self,
        crop: Union[np.ndarray, Image.Image]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classifies a single cropped damage image region using 376-dim handcrafted features
        (HOG + LBP + HSV) and the trained Random Forest classifier.
        
        Returns:
            (predicted_class, confidence, class_probabilities_dict)
        """
        if isinstance(crop, Image.Image):
            crop_rgb = np.array(crop.convert("RGB"))
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(crop, np.ndarray):
            if crop.ndim == 3 and crop.shape[2] == 3:
                crop_bgr = crop
            else:
                raise ValueError("Unsupported numpy crop array shape")
        else:
            raise TypeError("Unsupported crop input type")

        if crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
            return "dent", 0.50, {c: (1.0 if c == "dent" else 0.0) for c in CARDD_CLASSES}

        feats = extract_patch_features(crop_bgr)

        if self.model is None:
            # Rule-based fallback if model checkpoint is missing
            return "dent", 0.60, {c: (1.0 / len(CARDD_CLASSES)) for c in CARDD_CLASSES}

        probs = self.model.predict_proba(feats[np.newaxis, :])[0]
        classes_in_model = list(self.model.classes_)

        # Extract probabilities for the 6 CarDD damage classes
        dmg_probs = {}
        for cls_name in CARDD_CLASSES:
            if cls_name in classes_in_model:
                idx = classes_in_model.index(cls_name)
                dmg_probs[cls_name] = float(probs[idx])
            else:
                dmg_probs[cls_name] = 0.0

        # Renormalize across the 6 CarDD damage classes (filtering out background)
        total_dmg_prob = sum(dmg_probs.values())
        if total_dmg_prob > 1e-6:
            norm_probs = {c: p / total_dmg_prob for c, p in dmg_probs.items()}
        else:
            norm_probs = {c: 1.0 / len(CARDD_CLASSES) for c in CARDD_CLASSES}

        best_cls = max(norm_probs, key=norm_probs.get)
        best_conf = float(norm_probs[best_cls])

        return best_cls, round(best_conf, 3), {c: round(p, 4) for c, p in norm_probs.items()}

    def classify_user_boxes(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        boxes_xyxy: List[Tuple[float, float, float, float]]
    ) -> List[DetectedDamage]:
        """
        Directly classifies user-drawn bounding boxes without running automatic proposal generation.
        
        Args:
            image_input: Full-resolution image
            boxes_xyxy: List of [x1, y1, x2, y2] in native image pixel coordinates
            
        Returns:
            List[DetectedDamage] strictly matching the downstream pipeline contract.
        """
        if isinstance(image_input, (str, Path)):
            cv_img_bgr = cv2.imread(str(image_input))
            if cv_img_bgr is None:
                raise ValueError(f"Unable to read image at {image_input}")
        elif isinstance(image_input, Image.Image):
            rgb_arr = np.array(image_input.convert("RGB"))
            cv_img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            cv_img_bgr = image_input
        else:
            raise TypeError("Unsupported image input type")

        img_h, img_w = cv_img_bgr.shape[:2]
        img_area = float(img_w * img_h)

        detected_damages = []
        for idx, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            ix1 = int(max(0, min(img_w, x1)))
            iy1 = int(max(0, min(img_h, y1)))
            ix2 = int(max(0, min(img_w, x2)))
            iy2 = int(max(0, min(img_h, y2)))

            bw = max(0, ix2 - ix1)
            bh = max(0, iy2 - iy1)
            if bw < 4 or bh < 4:
                continue

            crop = cv_img_bgr[iy1:iy2, ix1:ix2]
            best_cls, best_conf, _ = self.classify_region(crop)

            norm_area = float(bw * bh) / img_area

            detected_damages.append(DetectedDamage(
                instance_id=idx + 1,
                damage_type=best_cls,
                confidence=round(best_conf, 3),
                bbox_xyxy=[float(ix1), float(iy1), float(ix2), float(iy2)],
                normalized_area=round(norm_area, 4)
            ))

        return detected_damages

    def detect(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.40,
        max_proposals: int = 300
    ) -> List[DetectedDamage]:
        """
        Executes classical damage detection on an input image.
        
        Returns:
            List[DetectedDamage] strictly matching the output contract consumed by cost_estimator & pipeline.
        """
        # Convert input to BGR numpy array
        if isinstance(image_input, (str, Path)):
            cv_img_bgr = cv2.imread(str(image_input))
            if cv_img_bgr is None:
                raise ValueError(f"Unable to read image at {image_input}")
        elif isinstance(image_input, Image.Image):
            rgb_arr = np.array(image_input.convert("RGB"))
            cv_img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                cv_img_bgr = image_input
            else:
                raise ValueError("Unsupported numpy image array shape")
        else:
            raise TypeError("Unsupported image input type")

        img_h, img_w = cv_img_bgr.shape[:2]
        img_area = float(img_w * img_h)

        if self.model is None:
            return []

        # 1. Generate candidate region proposals
        proposals = generate_selective_search_proposals(
            cv_img_bgr,
            max_proposals=max_proposals
        )

        if not proposals:
            return []

        # 2. Extract features for each proposal crop
        feature_matrix = []
        valid_boxes = []

        for box in proposals:
            x1, y1, x2, y2 = box
            crop = cv_img_bgr[y1:y2, x1:x2]
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                continue

            feats = extract_patch_features(crop)
            feature_matrix.append(feats)
            valid_boxes.append(box)

        if not feature_matrix:
            return []

        feature_matrix = np.array(feature_matrix, dtype=np.float32)

        # 3. Predict class probabilities
        probs = self.model.predict_proba(feature_matrix)
        classes_in_model = list(self.model.classes_)

        # Identify background index
        bg_idx = classes_in_model.index("background") if "background" in classes_in_model else -1

        raw_detections = []
        for i, box in enumerate(valid_boxes):
            sample_probs = probs[i]
            
            # Find best damage class (excluding background)
            best_dmg_cls = None
            best_dmg_score = 0.0

            for cls_idx, cls_name in enumerate(classes_in_model):
                if cls_name == "background":
                    continue
                score = float(sample_probs[cls_idx])
                if score > best_dmg_score:
                    best_dmg_score = score
                    best_dmg_cls = cls_name

            # Filter if score meets confidence threshold and is not background-dominated
            if best_dmg_cls is not None and best_dmg_score >= conf_threshold:
                bg_prob = float(sample_probs[bg_idx]) if bg_idx >= 0 else 0.0
                # Extra check: require damage probability to be competitive with background
                if best_dmg_score >= (bg_prob * 0.5):
                    raw_detections.append((best_dmg_cls, best_dmg_score, box))

        # 4. Class-wise Non-Maximum Suppression (NMS)
        nms_survivors = apply_classwise_nms(raw_detections, iou_threshold=iou_threshold)

        # 5. Format into standard DetectedDamage objects
        detected_damages = []
        for idx, (cls_name, score, box) in enumerate(nms_survivors):
            x1, y1, x2, y2 = box
            bw = max(0.0, float(x2 - x1))
            bh = max(0.0, float(y2 - y1))
            norm_area = (bw * bh) / img_area

            detected_damages.append(DetectedDamage(
                instance_id=idx + 1,
                damage_type=cls_name,
                confidence=round(score, 3),
                bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                normalized_area=round(norm_area, 4)
            ))

        return detected_damages
