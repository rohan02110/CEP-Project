"""
Classical Computer Vision & Tabular ML Vehicle Damage Assessment Pipeline.

Educational Note — 100% Classical & Interpretable Architecture:
---------------------------------------------------------------
This pipeline operates without any deep learning framework (no PyTorch, YOLO, or ResNet).
Data Flow:
1. Stage 0: Classical Vehicle RoI Isolation (Adaptive thresholding & contour centrality).
2. Stage 1: Classical Damage Candidate Proposals (Sobel gradient disturbances & morphology).
3. Stage 2: Handcrafted 119-dim Feature Extraction (Color moments, GLCM texture, LBP, Gabor banks, 2D FFT, Glare).
4. Stage 3: Tabular ML Severity & Damage Classification (Random Forest / XGBoost / Logistic Regression).
5. Stage 4: Specialized Micro-Analyzers (Tire circularity 4πA/P², Glass omnidirectional entropy & glare masking).
6. Stage 5: Knowledge-Grounded Actuarial Cost Engine & Monte Carlo Uncertainty Bounds.
7. Stage 6: Claim Fraud & Physical Inconsistency Auditing.
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Union, Any

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.config import (
    CARDD_CLASSES, CARDD_CLASS_TO_IDX, CARDD_IDX_TO_CLASS,
    SEVERITY_CLASSES, SEVERITY_CLASS_TO_IDX, SEVERITY_IDX_TO_CLASS,
    MODELS_DIR, PLOTS_DIR, METRICS_DIR, RANDOM_SEED
)
from src.features.preprocessing import preprocess_image, isolate_vehicle_roi, propose_damage_regions
from src.features.build_feature_table import extract_all_features_from_image
from src.glass_analyzer import GlassIntegrityAnalyzer, GlassDiagnosticResult
from src.cost_estimator import (
    VehicleProfile, VehicleSegment, DetectedDamage,
    ClaimCostEstimate, CostEstimationEngine
)
from src.classical_detector import ClassicalDamageDetector

# Vibrant color palette for bounding box annotations per damage class
CLASS_COLORS = {
    "dent": (41, 128, 185),          # Blue
    "scratch": (39, 174, 96),        # Green
    "crack": (230, 126, 34),         # Orange
    "glass shatter": (142, 68, 173), # Purple
    "lamp broken": (231, 76, 60),    # Red
    "tire flat": (52, 73, 94)        # Dark Slate
}


@dataclass
class FraudAnomalyReport:
    """Fraud and claim anomaly evaluation."""
    anomaly_score: float                  # [0.0, 1.0] (0 = Clean, 1 = High Risk)
    risk_level: str                       # "LOW", "MEDIUM", "HIGH"
    flags: List[str]                      # Specific heuristic trigger descriptions
    is_flagged_for_manual_audit: bool


@dataclass
class FullAssessmentResult:
    """Consolidated claim assessment report."""
    image_path: str
    vehicle_profile: VehicleProfile
    detected_damages: List[DetectedDamage]
    predicted_severity: str
    severity_probabilities: Dict[str, float]
    cost_estimate: ClaimCostEstimate
    fraud_audit: FraudAnomalyReport
    processing_time_ms: float
    primary_vehicle_roi: Optional[List[float]] = None
    filtered_background_damages_count: int = 0
    suppressed_tire_false_positives_count: int = 0
    suppressed_glass_false_positives_count: int = 0
    glass_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    annotated_image_path: Optional[str] = None
    extracted_features_count: int = 119


class ClassicalDamageAssessmentPipeline:
    """
    100% Classical Computer Vision & Tabular Machine Learning Pipeline.
    Replaces YOLOv8 and ResNet-50 with handcrafted classical features and tabular classifiers.
    """

    def __init__(
        self,
        severity_model_path: Optional[Union[str, Path]] = None,
        damage_model_path: Optional[Union[str, Path]] = None
    ):
        print("[PIPELINE INIT] Initializing Classical CV & Tabular ML Pipeline (CPU Native)...", flush=True)

        # 1. Load Classical Severity Classifier
        if severity_model_path is None:
            severity_model_path = MODELS_DIR / "classical_severity_best.joblib"

        if Path(severity_model_path).exists():
            print(f"[PIPELINE INIT] Loading Classical Severity Model: {severity_model_path}", flush=True)
            checkpoint = joblib.load(severity_model_path)
            self.severity_model = checkpoint["model"]
            self.severity_scaler = checkpoint.get("scaler", None)
            self.severity_features = checkpoint.get("feature_names", None)
            self.severity_model_name = checkpoint.get("model_name", "Classical Severity Classifier")
        else:
            print("[PIPELINE INIT] Classical severity model not yet trained. Using rule-based classical fallback.")
            self.severity_model = None
            self.severity_scaler = None
            self.severity_features = None
            self.severity_model_name = "Classical Heuristic Fallback"

        # 2. Load Classical Damage Detector (Selective Search + HOG/LBP/HSV + Random Forest)
        if damage_model_path is None:
            damage_model_path = MODELS_DIR / "classical_damage_rf_best.joblib"
            if not Path(damage_model_path).exists():
                fallback_path = MODELS_DIR / "classical_damage_best.joblib"
                if fallback_path.exists():
                    damage_model_path = fallback_path

        self.damage_detector = ClassicalDamageDetector(damage_model_path)
        self.damage_model = self.damage_detector.model
        self.damage_model_name = "Classical RF Detector (SS + HOG + LBP + HSV)" if self.damage_detector.model is not None else "Classical Selective Search Fallback"

        # 3. Knowledge-Grounded Actuarial Cost Engine
        self.cost_engine = CostEstimationEngine()

        # 4. Specialized Glass & Window Integrity Diagnostic Engine
        self.glass_analyzer = GlassIntegrityAnalyzer()
        print("[PIPELINE INIT] Classical Pipeline Ready!\n", flush=True)

    def locate_primary_subject_vehicle(
        self,
        rgb_img: np.ndarray
    ) -> Optional[List[float]]:
        """
        Isolates the central subject vehicle RoI using classical adaptive thresholding,
        Otsu segmentation, and contour centrality.
        """
        try:
            _, bbox = isolate_vehicle_roi(rgb_img)
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except Exception:
            return None

    def validate_tire_flat(
        self,
        cv_img_bgr: np.ndarray,
        bbox_xyxy: List[float],
        conf: float
    ) -> Tuple[bool, str]:
        """
        Geometrically and visually validates whether a wheel crop is a deflated/collapsed tire
        or an intact circular wheel using contour circularity (4πA/P²) and aspect ratio.
        """
        img_h, img_w = cv_img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(max(0, min(v, img_w if i % 2 == 0 else img_h))) for i, v in enumerate(bbox_xyxy)]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect_ratio = bw / float(bh)

        if 0.85 <= aspect_ratio <= 1.18 and conf < 0.85:
            crop = cv_img_bgr[y1:y2, x1:x2]
            if crop.shape[0] > 10 and crop.shape[1] > 10:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blur, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    max_c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(max_c)
                    perimeter = cv2.arcLength(max_c, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        if circularity > 0.45:
                            return False, f"Intact round wheel profile detected (circularity: {circularity:.2f}, aspect: {aspect_ratio:.2f})"

            return False, f"Normal inflated wheel profile (aspect ratio: {aspect_ratio:.2f})"

        return True, "Deflation/damage confirmed"

    def assess_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        vehicle_profile: Optional[VehicleProfile] = None,
        conf_threshold: float = 0.20,
        user_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
        user_damages: Optional[List[DetectedDamage]] = None,
        filter_background_vehicles: bool = True,
        validate_tires: bool = True,
        validate_glass: bool = True,
        glass_sensitivity: float = 0.50,
        output_plot_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> FullAssessmentResult:
        """
        Executes end-to-end 100% classical damage assessment on an input vehicle image.
        Supports direct user-drawn region classification and human-in-the-loop overrides.
        """
        start_time = time.perf_counter()

        # Resolve Image Input to PIL & Numpy RGB/BGR
        if isinstance(image_input, (str, Path)):
            img_path_str = str(image_input)
            pil_img = Image.open(image_input).convert("RGB")
            rgb_arr = np.array(pil_img)
            cv_img_bgr = cv2.imread(img_path_str)
        elif isinstance(image_input, Image.Image):
            img_path_str = "in_memory_image.jpg"
            pil_img = image_input.convert("RGB")
            rgb_arr = np.array(pil_img)
            cv_img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_path_str = "in_memory_array.jpg"
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                cv_img_bgr = image_input
                rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_arr)
            else:
                raise ValueError("Unsupported numpy image array shape")
        else:
            raise TypeError("Unsupported image input type")

        img_h, img_w = rgb_arr.shape[:2]
        img_area = float(img_w * img_h)

        if vehicle_profile is None:
            vehicle_profile = VehicleProfile(
                vehicle_id="AUTO-GEN-01",
                make_model="Insured Vehicle",
                year=2021,
                segment=VehicleSegment.MIDSIZE_SEDAN,
                actual_cash_value=2000000.0,
                deductible=25000.0
            )

        # -------------------------------------------------------------
        # STAGE 0: Classical Subject Vehicle RoI Isolation
        # -------------------------------------------------------------
        primary_roi = None
        if filter_background_vehicles and user_boxes is None and user_damages is None:
            primary_roi = self.locate_primary_subject_vehicle(rgb_arr)

        # -------------------------------------------------------------
        # STAGE 1: Damage Region Classification (User-Drawn or Classical RF)
        # -------------------------------------------------------------
        if user_damages is not None:
            raw_damages = user_damages
        elif user_boxes is not None and len(user_boxes) > 0:
            # User manually selected regions: classify each crop directly
            raw_damages = self.damage_detector.classify_user_boxes(cv_img_bgr, user_boxes)
        elif self.damage_detector.model is not None:
            raw_damages = self.damage_detector.detect(
                cv_img_bgr,
                conf_threshold=conf_threshold,
                iou_threshold=0.40,
                max_proposals=300
            )
        else:
            # Heuristic fallback if model checkpoint is not loaded
            proposals = propose_damage_regions(rgb_arr, max_regions=6)
            raw_damages = []
            for p_idx, prop in enumerate(proposals):
                bx1, by1, bx2, by2 = prop["bbox"]
                b_area_ratio = float(prop["area_ratio"])
                raw_damages.append(DetectedDamage(
                    instance_id=p_idx + 1,
                    damage_type="dent",
                    confidence=0.60,
                    bbox_xyxy=[float(bx1), float(by1), float(bx2), float(by2)],
                    normalized_area=round(b_area_ratio, 4)
                ))

        detected_damages: List[DetectedDamage] = []
        filtered_bg_count = 0
        suppressed_tire_count = 0
        suppressed_glass_count = 0
        glass_diagnostics: List[Dict[str, Any]] = []

        for dmg in raw_damages:
            x1, y1, x2, y2 = dmg.bbox_xyxy
            dmg_type = dmg.damage_type
            conf = dmg.confidence
            area_ratio = dmg.normalized_area

            # Subject vehicle ROI filter (reject background vehicles/distractors)
            if primary_roi is not None:
                dcx = (x1 + x2) / 2.0
                dcy = (y1 + y2) / 2.0
                vx1, vy1, vx2, vy2 = primary_roi
                if dcx < vx1 or dcx > vx2 or dcy < vy1 or dcy > vy2:
                    filtered_bg_count += 1
                    continue

            # Tire validation
            if dmg_type == "tire flat" and validate_tires:
                is_valid, _ = self.validate_tire_flat(cv_img_bgr, [x1, y1, x2, y2], conf)
                if not is_valid:
                    suppressed_tire_count += 1
                    continue

            # Glass validation
            if dmg_type == "glass shatter" and validate_glass:
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                crop_bgr = cv_img_bgr[max(0, iy1):min(img_h, iy2), max(0, ix1):min(img_w, ix2)]
                if crop_bgr.shape[0] >= 10 and crop_bgr.shape[1] >= 10:
                    diag = self.glass_analyzer.analyze_window_crop(crop_bgr, detector_conf=conf, sensitivity=glass_sensitivity)
                    if not diag.is_shattered_or_cracked:
                        suppressed_glass_count += 1
                        continue
                    else:
                        glass_diagnostics.append({
                            "instance_id": len(detected_damages) + 1,
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "status": diag.status,
                            "shatter_confidence": diag.shatter_confidence,
                            "fracture_density": diag.fracture_density,
                            "gradient_entropy": diag.gradient_entropy,
                            "laplacian_variance": diag.laplacian_variance,
                            "glare_ratio": diag.glare_ratio,
                            "reason": diag.reason,
                            "heatmap": diag.fracture_heatmap_rgb
                        })

            detected_damages.append(DetectedDamage(
                instance_id=len(detected_damages) + 1,
                damage_type=dmg_type,
                confidence=round(conf, 3),
                bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                normalized_area=round(area_ratio, 4)
            ))

        # -------------------------------------------------------------
        # STAGE 2: Global Classical Feature Extraction & Severity Prediction
        # -------------------------------------------------------------
        global_features = extract_all_features_from_image(rgb_arr)

        if self.severity_model is not None:
            feat_df = pd.DataFrame([global_features])
            if self.severity_features:
                feat_df = feat_df[self.severity_features]
            if self.severity_scaler is not None:
                feat_arr = self.severity_scaler.transform(feat_df)
            else:
                feat_arr = feat_df.values
            sev_probs = self.severity_model.predict_proba(feat_arr)[0]
            sev_idx = int(np.argmax(sev_probs))
            predicted_severity = SEVERITY_CLASSES[sev_idx]
            severity_prob_dict = {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(SEVERITY_CLASSES, sev_probs)
            }
            severity_conf = float(sev_probs[sev_idx])
        else:
            tot_area = sum(d.normalized_area for d in detected_damages)
            if len(detected_damages) == 0:
                predicted_severity = "normal"
                severity_conf = 0.85
            elif len(detected_damages) >= 3 or tot_area > 0.20:
                predicted_severity = "severe_crushed"
                severity_conf = 0.80
            else:
                predicted_severity = "moderate_breakage"
                severity_conf = 0.75

            severity_prob_dict = {
                "normal": 0.10 if predicted_severity != "normal" else 0.80,
                "moderate_breakage": 0.80 if predicted_severity == "moderate_breakage" else 0.10,
                "severe_crushed": 0.80 if predicted_severity == "severe_crushed" else 0.10
            }

        # -------------------------------------------------------------
        # STAGE 3: Fraud & Claim Inconsistency Auditing
        # -------------------------------------------------------------
        fraud_audit = self._audit_claim_fraud(
            detected_damages=detected_damages,
            predicted_severity=predicted_severity,
            severity_prob_dict=severity_prob_dict,
            vehicle_profile=vehicle_profile
        )

        # -------------------------------------------------------------
        # STAGE 4: Actuarial Cost Estimation
        # -------------------------------------------------------------
        cost_estimate = self.cost_engine.estimate_claim(
            vehicle=vehicle_profile,
            detected_damages=detected_damages,
            severity_class=predicted_severity,
            severity_prob=severity_conf
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FullAssessmentResult(
            image_path=img_path_str,
            vehicle_profile=vehicle_profile,
            detected_damages=detected_damages,
            predicted_severity=predicted_severity,
            severity_probabilities=severity_prob_dict,
            cost_estimate=cost_estimate,
            fraud_audit=fraud_audit,
            processing_time_ms=round(elapsed_ms, 2),
            primary_vehicle_roi=primary_roi,
            filtered_background_damages_count=filtered_bg_count,
            suppressed_tire_false_positives_count=suppressed_tire_count,
            suppressed_glass_false_positives_count=suppressed_glass_count,
            glass_diagnostics=glass_diagnostics,
            annotated_image_path=None,
            extracted_features_count=len(global_features)
        )

    def _audit_claim_fraud(
        self,
        detected_damages: List[DetectedDamage],
        predicted_severity: str,
        severity_prob_dict: Dict[str, float],
        vehicle_profile: VehicleProfile
    ) -> FraudAnomalyReport:
        """Applies insurance heuristic rules to detect claim anomalies."""
        flags = []
        anomaly_score = 0.0
        num_damages = len(detected_damages)
        tot_area = sum(d.normalized_area for d in detected_damages)

        if num_damages == 0 and predicted_severity in ["moderate_breakage", "severe_crushed"]:
            flags.append("[!] INCONSISTENCY: High crash severity predicted but zero localized damage candidate regions confirmed.")
            anomaly_score += 0.35

        if num_damages >= 4 and predicted_severity == "normal":
            flags.append("[!] INCONSISTENCY: Multiple damage instances detected (>3) but crash classified as 'normal'.")
            anomaly_score += 0.30

        if tot_area > 0.60:
            flags.append("[!] ANOMALY: Detected damages span over 60% of total image frame; inspect for multi-vehicle collision.")
            anomaly_score += 0.15

        anomaly_score = min(1.0, anomaly_score)
        if anomaly_score >= 0.40:
            risk_level = "HIGH"
            flagged = True
        elif anomaly_score >= 0.20:
            risk_level = "MEDIUM"
            flagged = False
        else:
            risk_level = "LOW"
            flagged = False

        return FraudAnomalyReport(
            anomaly_score=round(anomaly_score, 2),
            risk_level=risk_level,
            flags=flags,
            is_flagged_for_manual_audit=flagged
        )


# Backward compatibility alias
VehicleDamageAssessmentPipeline = ClassicalDamageAssessmentPipeline
