"""
End-to-End AI Vehicle Damage Assessment, Severity Classification,
Fraud/Anomaly Detection, Background Car Filtering, and Repair Cost Estimation Pipeline.
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
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

from src.config import (
    CARDD_CLASSES, CARDD_CLASS_TO_IDX, CARDD_IDX_TO_CLASS,
    SEVERITY_CLASSES, SEVERITY_CLASS_TO_IDX, SEVERITY_IDX_TO_CLASS,
    MODELS_DIR, PLOTS_DIR, METRICS_DIR, RANDOM_SEED
)
from src.feature_extractor import DeepFeatureExtractor
from src.cost_estimator import (
    VehicleProfile, VehicleSegment, DetectedDamage,
    ClaimCostEstimate, CostEstimationEngine
)

# Vibrant color palette for bounding box annotations per damage class
CLASS_COLORS = {
    "dent": (41, 128, 185),          # Blue
    "scratch": (39, 174, 96),        # Green
    "crack": (230, 126, 34),         # Orange
    "glass shatter": (142, 68, 173), # Purple
    "lamp broken": (231, 76, 60),    # Red
    "tire flat": (52, 73, 94)        # Dark Slate
}

# COCO Vehicle Class IDs in standard YOLOv8
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck


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
    annotated_image_path: Optional[str] = None


class VehicleDamageAssessmentPipeline:
    """
    Unified multi-stage inference pipeline:
    1. Primary Subject Vehicle Localization (filters background cars & shop clutter)
    2. Localized Damage Detection (YOLOv8) with False Positive Suppression
    3. Deep Feature Extraction (ResNet50) & Global Severity Classification
    4. Fraud & Claim Inconsistency Auditing
    5. Data-Driven & Actuarial Cost Estimation
    """

    def __init__(
        self,
        detector_weights: Optional[Union[str, Path]] = None,
        severity_model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PIPELINE INIT] Initializing End-to-End Pipeline on {self.device.upper()}...", flush=True)

        # 1. Load YOLOv8 Damage Detector
        if detector_weights is None:
            detector_weights = MODELS_DIR / "yolov8_damage_best.pt"
            if not detector_weights.exists():
                detector_weights = MODELS_DIR / "yolo_runs" / "cardd_yolov8n" / "weights" / "best.pt"
            if not detector_weights.exists():
                detector_weights = WORKSPACE_ROOT / "yolov8n.pt"

        print(f"[PIPELINE INIT] Loading YOLOv8 Damage Detector from: {detector_weights}", flush=True)
        self.detector = YOLO(str(detector_weights))

        # 2. Load Pretrained YOLO for Vehicle Localization (Subject RoI)
        coco_weights = WORKSPACE_ROOT / "yolov8n.pt"
        if not coco_weights.exists():
            coco_weights = "yolov8n.pt"
        print(f"[PIPELINE INIT] Initializing Vehicle Localization Model...", flush=True)
        try:
            self.vehicle_detector = YOLO(str(coco_weights))
        except Exception as e:
            print(f"[WARNING] Could not load vehicle detector: {e}")
            self.vehicle_detector = None

        # 3. Load ResNet50 Feature Extractor
        print("[PIPELINE INIT] Initializing ResNet50 Deep Feature Extractor...", flush=True)
        self.feature_extractor = DeepFeatureExtractor(device=self.device)

        # 4. Load Champion Severity Classifier
        if severity_model_path is None:
            severity_model_path = MODELS_DIR / "severity_classifier_best.joblib"

        print(f"[PIPELINE INIT] Loading Severity Classifier from: {severity_model_path}", flush=True)
        if Path(severity_model_path).exists():
            checkpoint = joblib.load(severity_model_path)
            self.severity_classifier = checkpoint["model"]
            self.severity_model_name = checkpoint.get("model_name", "Champion Classifier")
        else:
            print("[WARNING] Severity checkpoint not found! Using heuristic fallback.")
            self.severity_classifier = None
            self.severity_model_name = "Fallback Heuristic"

        # 5. Load Cost Estimation Engine (With ML Regressor)
        self.cost_engine = CostEstimationEngine()
        print("[PIPELINE INIT] Pipeline initialization complete!\n", flush=True)

    def locate_primary_subject_vehicle(
        self,
        pil_img: Image.Image,
        conf_thresh: float = 0.25
    ) -> Optional[List[float]]:
        """
        Detects all vehicles in the scene and selects the primary subject vehicle RoI
        based on foreground bounding area and center prominence.
        Returns: [x1, y1, x2, y2] of the primary vehicle or None.
        """
        if self.vehicle_detector is None:
            return None

        img_w, img_h = pil_img.size
        img_area = float(img_w * img_h)

        try:
            results = self.vehicle_detector.predict(
                source=pil_img,
                classes=COCO_VEHICLE_CLASSES,
                conf=conf_thresh,
                verbose=False,
                device=self.device
            )

            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                return None

            boxes = results[0].boxes
            best_roi = None
            best_score = -1.0

            for box in boxes:
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                bw = max(0.0, xyxy[2] - xyxy[0])
                bh = max(0.0, xyxy[3] - xyxy[1])
                area_ratio = (bw * bh) / img_area

                # Reject tiny vehicles (< 5% of frame) as background clutter
                if area_ratio < 0.05:
                    continue

                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                dist_center = np.sqrt(((cx - img_w / 2.0) / img_w) ** 2 + ((cy - img_h / 2.0) / img_h) ** 2)

                # Score combines size prominence with centrality
                score = area_ratio * (1.0 - 0.35 * dist_center)

                if score > best_score:
                    best_score = score
                    # Add a 4% margin around vehicle bounds to catch edge panel/lamp/bumper damage
                    pad_x = bw * 0.04
                    pad_y = bh * 0.04
                    best_roi = [
                        max(0.0, xyxy[0] - pad_x),
                        max(0.0, xyxy[1] - pad_y),
                        min(float(img_w), xyxy[2] + pad_x),
                        min(float(img_h), xyxy[3] + pad_y)
                    ]

            return best_roi
        except Exception as e:
            return None

    def validate_tire_flat(
        self,
        cv_img_bgr: np.ndarray,
        bbox_xyxy: List[float],
        conf: float
    ) -> Tuple[bool, str]:
        """
        Geometrically and visually validates whether a detected wheel is genuinely flat/punctured
        or an intact, normal inflated tire.
        Returns: (is_valid_flat, reason_string)
        """
        # 1. Require elevated minimum confidence threshold for tire flat
        if conf < 0.45:
            return False, f"Confidence ({conf*100:.1f}%) below minimum required threshold for flat tire (45%)"

        img_h, img_w = cv_img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(max(0, min(v, img_w if i % 2 == 0 else img_h))) for i, v in enumerate(bbox_xyxy)]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        aspect_ratio = bw / float(bh)

        # 2. Geometric Shape Analysis:
        # A normal, fully inflated wheel is nearly circular / square bounding box (aspect ratio 0.85 - 1.18).
        # A deflated flat tire has collapsed under car weight, flattening at contact surface (aspect ratio > 1.25).
        if 0.85 <= aspect_ratio <= 1.18 and conf < 0.85:
            # Check edge gradient distribution
            crop = cv_img_bgr[y1:y2, x1:x2]
            if crop.shape[0] > 10 and crop.shape[1] > 10:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Compute contour roundness of rim
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
        iou_threshold: float = 0.45,
        filter_background_vehicles: bool = True,
        validate_tires: bool = True,
        min_conf_per_class: Optional[Dict[str, float]] = None,
        output_plot_path: Optional[Union[str, Path]] = None
    ) -> FullAssessmentResult:
        """
        Executes end-to-end multi-stage assessment on an input vehicle image.
        """
        start_time = time.perf_counter()

        # Class-specific confidence thresholds
        default_class_confs = {
            "dent": conf_threshold,
            "scratch": conf_threshold,
            "crack": max(0.20, conf_threshold),
            "glass shatter": max(0.25, conf_threshold),
            "lamp broken": max(0.25, conf_threshold),
            "tire flat": max(0.48, conf_threshold) # Higher threshold to prevent normal wheel false positives
        }
        if min_conf_per_class:
            default_class_confs.update(min_conf_per_class)

        # Resolve Image Input to PIL & Numpy BGR
        if isinstance(image_input, (str, Path)):
            img_path_str = str(image_input)
            pil_img = Image.open(image_input).convert("RGB")
            cv_img_bgr = cv2.imread(img_path_str)
        elif isinstance(image_input, Image.Image):
            img_path_str = "in_memory_image.jpg"
            pil_img = image_input.convert("RGB")
            cv_img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_path_str = "in_memory_array.jpg"
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                cv_img_bgr = image_input
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Unsupported numpy image array shape")
        else:
            raise TypeError("Unsupported image input type")

        img_w, img_h = pil_img.size
        img_area = float(img_w * img_h)

        if vehicle_profile is None:
            vehicle_profile = VehicleProfile(
                vehicle_id="AUTO-GEN-01",
                make_model="Insured Vehicle",
                year=2021,
                segment=VehicleSegment.MIDSIZE_SEDAN,
                actual_cash_value=20000.0,
                deductible=500.0
            )

        # -------------------------------------------------------------
        # STAGE 0: Primary Subject Vehicle RoI Localization
        # -------------------------------------------------------------
        primary_roi = None
        if filter_background_vehicles:
            primary_roi = self.locate_primary_subject_vehicle(pil_img)

        # -------------------------------------------------------------
        # STAGE 1: Localized Damage Detection (YOLOv8) & Intelligent Filtering
        # -------------------------------------------------------------
        # We query YOLO with lowest common denominator threshold, then filter per-class and per-RoI
        query_conf = min(conf_threshold, 0.15)
        det_results = self.detector.predict(
            source=pil_img,
            conf=query_conf,
            iou=iou_threshold,
            verbose=False,
            device=self.device
        )

        detected_damages: List[DetectedDamage] = []
        filtered_bg_count = 0
        suppressed_tire_count = 0

        if len(det_results) > 0 and det_results[0].boxes is not None:
            boxes = det_results[0].boxes
            raw_candidates = []

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                cls_name = CARDD_CLASSES[cls_id] if cls_id < len(CARDD_CLASSES) else "dent"
                raw_candidates.append((cls_name, conf, xyxy))

            # Filter candidates
            for cls_name, conf, xyxy in raw_candidates:
                # 1. Per-Class Minimum Confidence Filter
                min_c = default_class_confs.get(cls_name, conf_threshold)
                if conf < min_c:
                    continue

                # 2. Primary Subject Vehicle RoI Filter (Background Car & Clutter Elimination)
                if primary_roi is not None:
                    # Calculate center point of damage bounding box
                    dcx = (xyxy[0] + xyxy[2]) / 2.0
                    dcy = (xyxy[1] + xyxy[3]) / 2.0
                    vx1, vy1, vx2, vy2 = primary_roi

                    # Check if damage center lies inside primary vehicle RoI
                    if dcx < vx1 or dcx > vx2 or dcy < vy1 or dcy > vy2:
                        filtered_bg_count += 1
                        continue

                # 3. Tire Flat False Positive Verification Filter
                if cls_name == "tire flat" and validate_tires:
                    is_valid, reason = self.validate_tire_flat(cv_img_bgr, xyxy, conf)
                    if not is_valid:
                        suppressed_tire_count += 1
                        continue

                bw = max(0.0, xyxy[2] - xyxy[0])
                bh = max(0.0, xyxy[3] - xyxy[1])
                norm_area = (bw * bh) / img_area

                detected_damages.append(DetectedDamage(
                    instance_id=len(detected_damages) + 1,
                    damage_type=cls_name,
                    confidence=round(conf, 3),
                    bbox_xyxy=[round(c, 1) for c in xyxy],
                    normalized_area=round(norm_area, 4)
                ))

        # -------------------------------------------------------------
        # STAGE 2: Deep Feature Extraction & Severity Classification
        # -------------------------------------------------------------
        cnn_features = self.feature_extractor.extract_image_embedding(pil_img)
        if len(cnn_features.shape) == 1:
            cnn_features = cnn_features.reshape(1, -1)

        if self.severity_classifier is not None:
            sev_probs = self.severity_classifier.predict_proba(cnn_features)[0]
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
                severity_conf = 0.90
            elif len(detected_damages) >= 4 or tot_area > 0.25:
                predicted_severity = "severe_crushed"
                severity_conf = 0.85
            else:
                predicted_severity = "moderate_breakage"
                severity_conf = 0.80

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
        # STAGE 4: Actuarial Cost Estimation & Uncertainty Analysis
        # -------------------------------------------------------------
        cost_estimate = self.cost_engine.estimate_claim(
            vehicle=vehicle_profile,
            detected_damages=detected_damages,
            severity_class=predicted_severity,
            severity_prob=severity_conf
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # -------------------------------------------------------------
        # STAGE 5: Render Visual Annotated Damage Inspection Sheet
        # -------------------------------------------------------------
        annotated_path_str = None
        if output_plot_path is not None:
            self._render_full_inspection_sheet(
                cv_img_bgr=cv_img_bgr,
                detected_damages=detected_damages,
                primary_roi=primary_roi,
                predicted_severity=predicted_severity,
                severity_conf=severity_conf,
                cost_estimate=cost_estimate,
                fraud_audit=fraud_audit,
                elapsed_ms=elapsed_ms,
                save_path=Path(output_plot_path)
            )
            annotated_path_str = str(output_plot_path)

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
            annotated_image_path=annotated_path_str
        )

    def _audit_claim_fraud(
        self,
        detected_damages: List[DetectedDamage],
        predicted_severity: str,
        severity_prob_dict: Dict[str, float],
        vehicle_profile: VehicleProfile
    ) -> FraudAnomalyReport:
        """
        Applies insurance heuristic rules to detect claim anomalies and discrepancies.
        """
        flags = []
        anomaly_score = 0.0
        num_damages = len(detected_damages)
        tot_area = sum(d.normalized_area for d in detected_damages)

        if num_damages == 0 and predicted_severity in ["moderate_breakage", "severe_crushed"]:
            flags.append("[!] INCONSISTENCY: High crash severity predicted but zero localized damage bounding boxes detected.")
            anomaly_score += 0.35

        if num_damages >= 4 and predicted_severity == "normal":
            flags.append("[!] INCONSISTENCY: Multiple damage instances detected (>3) but crash classified as 'normal'.")
            anomaly_score += 0.30

        if num_damages > 0:
            avg_conf = np.mean([d.confidence for d in detected_damages])
            if avg_conf < 0.45:
                flags.append("[!] QUALITY: Low detection confidence (<45%); potential image blur or lighting anomaly.")
                anomaly_score += 0.20

        if tot_area > 0.60:
            flags.append("[!] ANOMALY: Detected damages span over 60% of total image frame; inspect for multi-vehicle collision.")
            anomaly_score += 0.15

        anomaly_score = min(1.0, anomaly_score)
        if anomaly_score >= 0.40:
            risk_level = "HIGH"
            flagged_for_audit = True
        elif anomaly_score >= 0.20:
            risk_level = "MEDIUM"
            flagged_for_audit = False
        else:
            risk_level = "LOW"
            flagged_for_audit = False

        return FraudAnomalyReport(
            anomaly_score=round(anomaly_score, 2),
            risk_level=risk_level,
            flags=flags,
            is_flagged_for_manual_audit=flagged_for_audit
        )

    def _render_full_inspection_sheet(
        self,
        cv_img_bgr: np.ndarray,
        detected_damages: List[DetectedDamage],
        primary_roi: Optional[List[float]],
        predicted_severity: str,
        severity_conf: float,
        cost_estimate: ClaimCostEstimate,
        fraud_audit: FraudAnomalyReport,
        elapsed_ms: float,
        save_path: Path
    ):
        """
        Renders a composite inspection sheet.
        """
        annotated_bgr = cv_img_bgr.copy()
        img_h, img_w, _ = annotated_bgr.shape

        # Draw Primary Subject Vehicle RoI boundary if detected
        if primary_roi is not None:
            vx1, vy1, vx2, vy2 = [int(v) for v in primary_roi]
            cv2.rectangle(annotated_bgr, (vx1, vy1), (vx2, vy2), (0, 215, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(
                annotated_bgr, "PRIMARY SUBJECT VEHICLE RoI",
                (vx1 + 5, max(20, vy1 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA
            )

        # Draw Damage Bounding Boxes
        for dmg in detected_damages:
            x1, y1, x2, y2 = [int(v) for v in dmg.bbox_xyxy]
            color = CLASS_COLORS.get(dmg.damage_type, (0, 255, 0))
            bgr_color = (color[2], color[1], color[0])

            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), bgr_color, 3)

            label_text = f"#{dmg.instance_id} {dmg.damage_type.upper()} ({dmg.confidence*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_bgr, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), bgr_color, -1)
            cv2.putText(
                annotated_bgr, label_text, (x1 + 5, max(15, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], width_ratios=[1.2, 1.0])

        ax_img = fig.add_subplot(gs[0, :])
        ax_img.imshow(annotated_rgb)
        ax_img.set_title(
            f"AI Vision Damage Localization ({len(detected_damages)} Damage Instances on Subject Vehicle) | Latency: {elapsed_ms:.1f} ms",
            fontsize=13, fontweight="bold", pad=10
        )
        ax_img.axis("off")

        ax_bar = fig.add_subplot(gs[1, 0])
        categories = ["Body Labor", "Paint Labor", "Mech Labor", "OEM Parts", "Paint Mats", "Structural", "Supplies"]
        values = [
            cost_estimate.total_body_labor_cost,
            cost_estimate.total_paint_labor_cost,
            cost_estimate.total_mech_labor_cost,
            cost_estimate.total_parts_cost,
            cost_estimate.total_paint_materials_cost,
            cost_estimate.structural_overhead_cost,
            cost_estimate.shop_supplies_fee
        ]
        colors = ["#2b5c8f", "#3a7bd5", "#43a047", "#e53935", "#fb8c00", "#8e24aa", "#757575"]
        bars = ax_bar.bar(categories, values, color=colors, edgecolor="black", linewidth=0.8)
        ax_bar.set_title("Itemized Repair Cost Breakdown", fontsize=11, fontweight="bold")
        ax_bar.set_ylabel("USD ($)", fontsize=10, fontweight="bold")
        ax_bar.set_xticks(range(len(categories)))
        ax_bar.set_xticklabels(categories, rotation=20, ha="right", fontsize=9)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

        for b in bars:
            h = b.get_height()
            if h > 0:
                ax_bar.annotate(f"${h:,.0f}", xy=(b.get_x() + b.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax_card = fig.add_subplot(gs[1, 1])
        ax_card.axis("off")

        v = cost_estimate.vehicle
        card_text = (
            f"ASSESSMENT & ADJUDICATION SUMMARY\n"
            f"------------------------------------------------------------------\n"
            f"• Vehicle: {v.year} {v.make_model} [{v.segment.upper()}]\n"
            f"• Market Value (ACV): USD {v.actual_cash_value:,.2f} | Deductible: USD {v.deductible:,.2f}\n"
            f"• Severity: {predicted_severity.upper()} (Confidence: {severity_conf*100:.1f}%)\n"
            f"• Gross Repair Cost: USD {cost_estimate.estimated_total_cost:,.2f}\n"
            f"• Net Insurer Payout: USD {cost_estimate.net_claim_payout:,.2f}\n"
            f"• ML Regressor Benchmark: USD {cost_estimate.ml_empirical_estimate:,.2f}\n"
            f"• 90% Confidence Bounds: USD {cost_estimate.cost_p10_optimistic:,.0f} - USD {cost_estimate.cost_p90_pessimistic:,.0f}\n"
            f"• Loss Ratio: {cost_estimate.loss_ratio*100:.1f}% of ACV (CTL Threshold: 75%)\n"
            f"• Claim Status: {'[!] CONSTRUCTIVE TOTAL LOSS' if cost_estimate.is_total_loss else '[OK] REPAIR AUTHORIZED'}\n"
            f"• Fraud / Anomaly Risk: {fraud_audit.risk_level} (Score: {fraud_audit.anomaly_score:.2f})\n"
        )
        if fraud_audit.flags:
            card_text += f"• Flags: {fraud_audit.flags[0][:60]}...\n"

        bbox_props = dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#2b5c8f", linewidth=2.0)
        ax_card.text(0.05, 0.50, card_text, fontsize=10.5, fontfamily="monospace",
                     verticalalignment="center", bbox=bbox_props)

        plt.suptitle(
            f"AI Motor Insurance Claim Assessment Report — Vehicle Ref: {v.vehicle_id}\n"
            f"Severity: {predicted_severity.upper()} | Estimated Claim Payout: USD {cost_estimate.net_claim_payout:,.2f}",
            fontsize=14, fontweight="bold", y=0.98
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[PIPELINE] Saved Complete Inspection Sheet to: {save_path}", flush=True)


def test_pipeline_on_sample_images():
    """
    Runs the full end-to-end pipeline on several unseen test images from CarDD
    and outputs visual inspection sheets.
    """
    print("\n" + "="*70)
    print("STARTING END-TO-END PIPELINE VALIDATION ON TEST DATASET")
    print("="*70, flush=True)

    pipeline = VehicleDamageAssessmentPipeline()

    test_img_dir = WORKSPACE_ROOT / "data" / "processed" / "cardd" / "images" / "test"
    assert test_img_dir.exists(), f"Test image directory not found: {test_img_dir}"

    test_images = sorted(list(test_img_dir.glob("*.jpg")))[:4]
    print(f"Testing pipeline on {len(test_images)} held-out CarDD test images...\n", flush=True)

    vehicles = [
        VehicleProfile("CLM-TST-101", "Honda Civic EX", 2022, VehicleSegment.MIDSIZE_SEDAN, 24000.0, 500.0),
        VehicleProfile("CLM-TST-102", "Toyota RAV4 XLE", 2021, VehicleSegment.SUV_CROSSOVER, 28500.0, 500.0),
        VehicleProfile("CLM-TST-103", "BMW 330i Sport", 2020, VehicleSegment.LUXURY_PREMIUM, 35000.0, 1000.0),
        VehicleProfile("CLM-TST-104", "Hyundai Elantra", 2018, VehicleSegment.ECONOMY, 9500.0, 300.0)
    ]

    results_summary = []

    for idx, (img_p, veh) in enumerate(zip(test_images, vehicles)):
        out_plot = PLOTS_DIR / f"pipeline_test_inspection_{img_p.stem}.png"
        print(f"[{idx+1}/{len(test_images)}] Assessing Image: {img_p.name} for {veh.make_model}...")
        
        result = pipeline.assess_image(
            image_input=img_p,
            vehicle_profile=veh,
            conf_threshold=0.20,
            filter_background_vehicles=True,
            validate_tires=True,
            output_plot_path=out_plot
        )

        print(f"  -> Detections: {len(result.detected_damages)} damage instance(s)")
        print(f"  -> Filtered BG Damages: {result.filtered_background_damages_count}, Suppressed Tires: {result.suppressed_tire_false_positives_count}")
        for d in result.detected_damages:
            print(f"     * #{d.instance_id} {d.damage_type} (conf: {d.confidence*100:.1f}%, norm_area: {d.normalized_area:.4f})")
        print(f"  -> Severity: {result.predicted_severity.upper()} ({result.severity_probabilities})")
        print(f"  -> Gross Repair Cost: ${result.cost_estimate.estimated_total_cost:,.2f} | ML Benchmark: ${result.cost_estimate.ml_empirical_estimate:,.2f}")
        print(f"  -> Net Payout: ${result.cost_estimate.net_claim_payout:,.2f}")
        print(f"  -> Fraud/Anomaly Risk: {result.fraud_audit.risk_level} (Score: {result.fraud_audit.anomaly_score})")
        print(f"  -> Latency: {result.processing_time_ms:.1f} ms\n")

        results_summary.append({
            "image": img_p.name,
            "vehicle": veh.make_model,
            "segment": veh.segment.value,
            "damages_detected": len(result.detected_damages),
            "damage_classes": [d.damage_type for d in result.detected_damages],
            "severity": result.predicted_severity,
            "gross_repair_cost": result.cost_estimate.estimated_total_cost,
            "ml_empirical_estimate": result.cost_estimate.ml_empirical_estimate,
            "net_payout": result.cost_estimate.net_claim_payout,
            "is_total_loss": result.cost_estimate.is_total_loss,
            "fraud_risk": result.fraud_audit.risk_level,
            "latency_ms": result.processing_time_ms,
            "inspection_sheet": str(out_plot)
        })

    summary_path = METRICS_DIR / "pipeline_test_benchmark.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"[SUCCESS] All pipeline test runs completed! Benchmark saved to: {summary_path}")
    return results_summary


if __name__ == "__main__":
    test_pipeline_on_sample_images()
