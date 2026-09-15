"""
Interactive AI Vehicle Damage Assessment, Inspection & Actuarial Repair Cost Estimator.
Streamlit Web Dashboard for Automated Insurance Claim Adjudication.
Features:
- User-drawn rectangular damage region selection via streamlit-drawable-canvas.
- Native resolution coordinate mapping & cropping.
- 100% Classical ML classification (HOG + LBP + HSV -> Random Forest).
- Human-in-the-loop class override with 100% confidence calibration.
- Full actuarial repair cost valuation in Indian Rupees (INR / ₹).
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Tuple

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas

from src.config import (
    CARDD_CLASSES, SEVERITY_CLASSES, PROCESSED_DATA_DIR,
    USD_TO_INR, CURRENCY_SYMBOL
)
from src.cost_estimator import (
    VehicleProfile, VehicleSegment, ClaimCostEstimate, DetectedDamage
)
from src.pipeline import (
    VehicleDamageAssessmentPipeline, CLASS_COLORS, FullAssessmentResult
)
from src.glass_analyzer import GlassIntegrityAnalyzer, GlassDiagnosticResult

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="AI Vehicle Damage & Repair Cost Estimator (INR)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN DARK MODE CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #262c3a 100%);
        padding: 18px 22px;
        border-radius: 12px;
        border: 1px solid #333d4f;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    .metric-title {
        color: #9aa0a6;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 12px;
        font-weight: 500;
        margin-top: 4px;
    }
    .total-loss-banner {
        background: linear-gradient(90deg, rgba(231,76,60,0.2) 0%, rgba(192,57,43,0.1) 100%);
        border-left: 5px solid #e74c3c;
        padding: 14px 18px;
        border-radius: 6px;
        color: #ff6b6b;
        font-weight: 600;
        margin: 15px 0;
    }
    .repairable-banner {
        background: linear-gradient(90deg, rgba(46,204,113,0.2) 0%, rgba(39,174,96,0.1) 100%);
        border-left: 5px solid #2ecc71;
        padding: 14px 18px;
        border-radius: 6px;
        color: #2ecc71;
        font-weight: 600;
        margin: 15px 0;
    }
    .canvas-instruction-card {
        background: linear-gradient(90deg, rgba(41,128,185,0.15) 0%, rgba(31,58,82,0.1) 100%);
        border-left: 4px solid #3498db;
        padding: 12px 16px;
        border-radius: 8px;
        color: #d6eaf8;
        font-size: 14px;
        margin-bottom: 15px;
    }
    .override-card {
        background: #1a1e28;
        border: 1px solid #2e384d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


def get_pipeline():
    """Initializes the end-to-end assessment pipeline with fresh module definitions."""
    import importlib
    import src.classical_detector
    import src.cost_estimator
    import src.pipeline
    importlib.reload(src.classical_detector)
    importlib.reload(src.cost_estimator)
    importlib.reload(src.pipeline)
    return src.pipeline.VehicleDamageAssessmentPipeline()


def annotate_image_pil(
    image: Image.Image,
    damages: List[DetectedDamage],
    primary_roi: list = None
) -> Image.Image:
    """Draws color-coded bounding boxes and label badges onto a PIL Image."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    for dmg in damages:
        x1, y1, x2, y2 = dmg.bbox_xyxy
        color = CLASS_COLORS.get(dmg.damage_type, (41, 128, 185))
        
        # Bounding Box outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Label Badge Header
        conf_text = "100% (Human)" if dmg.confidence >= 0.999 else f"{dmg.confidence*100:.0f}%"
        label = f"#{dmg.instance_id} {dmg.damage_type.upper()} ({conf_text})"
        text_bbox = draw.textbbox((x1, y1), label)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        
        badge_y0 = max(0, y1 - th - 8)
        draw.rectangle([x1, badge_y0, x1 + tw + 12, y1], fill=color)
        draw.text((x1 + 6, badge_y0 + 3), label, fill=(255, 255, 255))
        
    return annotated


def main():
    st.title("🚗 AI Vehicle Damage Assessment & Valuation")
    st.markdown("**User-Guided Damage Region Selection, Classical ML Classification (HOG+LBP+HSV), and Actuarial Cost Estimator in Indian Rupees (₹)**")

    pipeline = get_pipeline()

    # --- SIDEBAR: VEHICLE & POLICY CONFIGURATION ---
    st.sidebar.header("📋 Vehicle & Policy Details")
    make_model = st.sidebar.text_input("Vehicle Make & Model", value="BMW M3")
    year = st.sidebar.number_input("Model Year", min_value=2000, max_value=2026, value=2021)
    
    segment_choice = st.sidebar.selectbox(
        "Vehicle Category",
        options=["Economy / Compact", "Midsize Sedan", "SUV / Crossover", "Luxury / Premium"],
        index=3
    )
    segment_map = {
        "Economy / Compact": VehicleSegment.ECONOMY,
        "Midsize Sedan": VehicleSegment.MIDSIZE_SEDAN,
        "SUV / Crossover": VehicleSegment.SUV_CROSSOVER,
        "Luxury / Premium": VehicleSegment.LUXURY_PREMIUM
    }
    selected_segment = segment_map[segment_choice]

    acv = st.sidebar.number_input(
        f"Actual Cash Value (ACV {CURRENCY_SYMBOL})",
        min_value=50000.0,
        max_value=20000000.0,
        value=4500000.0,
        step=50000.0,
        help="Market value of the vehicle in Indian Rupees."
    )
    deductible = st.sidebar.number_input(
        f"Policy Deductible ({CURRENCY_SYMBOL})",
        min_value=0.0,
        max_value=500000.0,
        value=25000.0,
        step=5000.0,
        help="Policyholder deductible payable upon claim approval in Indian Rupees."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🖼️ Test Dataset Presets")
    test_img_dir = PROCESSED_DATA_DIR / "cardd" / "images" / "test"
    sample_files = []
    if test_img_dir.exists():
        sample_files = sorted([f.name for f in test_img_dir.glob("*.jpg")])[:15]

    sample_select = st.sidebar.selectbox(
        "Pick a sample from held-out test set:",
        options=["-- None (Upload my own) --"] + sample_files,
        index=0
    )

    # --- IMAGE SELECTION / UPLOAD ---
    col_up1, col_up2 = st.columns([2, 1])

    with col_up1:
        uploaded_file = st.file_uploader("Upload Vehicle Damage Image", type=["jpg", "jpeg", "png"])

    selected_image = None
    image_name = "Uploaded Vehicle"

    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name
    elif sample_select != "-- None (Upload my own) --":
        sample_path = test_img_dir / sample_select
        if sample_path.exists():
            selected_image = Image.open(sample_path).convert("RGB")
            image_name = sample_select

    if selected_image is None:
        st.info("👆 Please upload a vehicle photo or select a sample image from the left sidebar to begin.")
        return

    # Native image resolution
    orig_w, orig_h = selected_image.size
    img_np = np.array(selected_image)
    cv_img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Build Vehicle Profile (Natively in INR)
    vehicle_profile = VehicleProfile(
        vehicle_id="CLM-" + str(int(time.time()))[-6:],
        make_model=make_model,
        year=year,
        segment=selected_segment,
        actual_cash_value=acv,
        deductible=deductible
    )

    # Session State Initialization for Interactive Drawing & Overrides
    if "current_image_name" not in st.session_state or st.session_state["current_image_name"] != image_name:
        st.session_state["current_image_name"] = image_name
        st.session_state["confirmed_damages"] = []
        st.session_state["overrides"] = {}

    st.markdown("---")
    st.markdown("### 🎯 Interactive Damage Region Selection")
    st.markdown(
        '<div class="canvas-instruction-card">📐 <b>Instructions:</b> Use your mouse to draw rectangular bounding boxes directly over damaged parts (dents, scratches, cracks, broken lamps, shattered windows, flat tires). You can draw multiple boxes. When finished, click <b>Confirm Damage Selections & Run AI Classification</b>.</div>',
        unsafe_allow_html=True
    )

    # Calculate optimal canvas display dimensions maintaining aspect ratio
    canvas_max_w = 720
    scale_factor = min(1.0, canvas_max_w / float(orig_w))
    disp_w = int(orig_w * scale_factor)
    disp_h = int(orig_h * scale_factor)

    display_img = selected_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

    col_canvas, col_panel = st.columns([1.3, 1])

    with col_canvas:
        st.markdown(f"**Drawing Canvas** (Native Resolution: {orig_w}×{orig_h}px | Canvas Display: {disp_w}×{disp_h}px)")
        canvas_result = st_canvas(
            fill_color="rgba(255, 75, 75, 0.25)",
            stroke_width=2,
            stroke_color="#ff4b4b",
            background_image=display_img,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key=f"canvas_{image_name}",
        )

        c_btn1, c_btn2 = st.columns([1.5, 1])
        with c_btn1:
            confirm_clicked = st.button("🔍 Confirm Damage Selections & Run AI Classification", type="primary")
        with c_btn2:
            if st.button("🔄 Reset Selections"):
                st.session_state["confirmed_damages"] = []
                st.session_state["overrides"] = {}
                st.rerun()

    # Extract Drawn Rectangles / Regions with Multi-Tier Fallback
    def parse_canvas_boxes(canvas_obj, orig_width: int, orig_height: int, disp_width: int, disp_height: int) -> List[Tuple[int, int, int, int]]:
        sx = orig_width / float(disp_width)
        sy = orig_height / float(disp_height)
        boxes: List[Tuple[int, int, int, int]] = []

        if canvas_obj is None:
            return boxes

        # Tier 1: Parse JSON data objects from fabric.js
        if canvas_obj.json_data is not None and isinstance(canvas_obj.json_data, dict):
            objects = canvas_obj.json_data.get("objects", [])
            for obj in objects:
                left = float(obj.get("left", 0.0))
                top = float(obj.get("top", 0.0))
                scale_x_obj = float(obj.get("scaleX", 1.0))
                scale_y_obj = float(obj.get("scaleY", 1.0))
                w_box = float(obj.get("width", 0.0)) * scale_x_obj
                h_box = float(obj.get("height", 0.0)) * scale_y_obj

                if obj.get("flipX"):
                    w_box = -abs(w_box)
                if obj.get("flipY"):
                    h_box = -abs(h_box)

                b_left = min(left, left + w_box)
                b_top = min(top, top + h_box)
                b_right = max(left, left + w_box)
                b_bottom = max(top, top + h_box)

                # If path or polygon was drawn
                if "path" in obj and isinstance(obj["path"], list):
                    pts_x = [pt[1] for pt in obj["path"] if len(pt) > 1 and isinstance(pt[1], (int, float))]
                    pts_y = [pt[2] for pt in obj["path"] if len(pt) > 2 and isinstance(pt[2], (int, float))]
                    if pts_x and pts_y:
                        b_left = min(pts_x)
                        b_top = min(pts_y)
                        b_right = max(pts_x)
                        b_bottom = max(pts_y)

                x1 = int(max(0, min(orig_width, b_left * sx)))
                y1 = int(max(0, min(orig_height, b_top * sy)))
                x2 = int(max(0, min(orig_width, b_right * sx)))
                y2 = int(max(0, min(orig_height, b_bottom * sy)))

                if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                    boxes.append((x1, y1, x2, y2))

        # Tier 2: Fallback to image_data alpha channel contour detection if enabled/available
        if not boxes:
            try:
                img_arr = getattr(canvas_obj, "image_data", None)
                if img_arr is not None and isinstance(img_arr, np.ndarray) and img_arr.ndim == 3 and img_arr.shape[2] == 4:
                    alpha = img_arr[:, :, 3]
                    if np.any(alpha > 10):
                        mask = (alpha > 10).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            cx, cy, cw, ch = cv2.boundingRect(cnt)
                            if cw >= 4 and ch >= 4:
                                x1 = int(max(0, min(orig_width, cx * sx)))
                                y1 = int(max(0, min(orig_height, cy * sy)))
                                x2 = int(max(0, min(orig_width, (cx + cw) * sx)))
                                y2 = int(max(0, min(orig_height, (cy + ch) * sy)))
                                if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                                    boxes.append((x1, y1, x2, y2))
            except (RuntimeError, AttributeError, Exception):
                pass

        return boxes

    # Live Box Count Notification
    detected_live_boxes = parse_canvas_boxes(canvas_result, orig_w, orig_h, disp_w, disp_h)
    if detected_live_boxes and not st.session_state.get("confirmed_damages"):
        st.caption(f"📌 **{len(detected_live_boxes)} region(s) currently marked on canvas.** Click 'Confirm Damage Selections' below to run AI classification.")

    # Process Drawn Rectangles on Confirmation
    if confirm_clicked:
        parsed_boxes = parse_canvas_boxes(canvas_result, orig_w, orig_h, disp_w, disp_h)

        if not parsed_boxes:
            st.warning("⚠️ No bounding boxes drawn! Please draw at least one rectangle over the damaged area on the canvas.")
        else:
            with st.spinner("Extracting 376-dim handcrafted features (HOG+LBP+HSV) and classifying regions..."):
                classified_damages: List[DetectedDamage] = []
                for idx, (x1, y1, x2, y2) in enumerate(parsed_boxes):
                    # Crop patch directly from native original resolution image
                    crop_bgr = cv_img_bgr[y1:y2, x1:x2]
                    
                    # Compute exact normalized surface area a_k from user-drawn coordinates
                    norm_area = float((x2 - x1) * (y2 - y1)) / float(orig_w * orig_h)

                    # 100% Classical HOG + LBP + HSV -> Random Forest Classification with Area Prior
                    pred_cls, pred_conf, _ = pipeline.damage_detector.classify_region(crop_bgr, area_ratio=norm_area)

                    classified_damages.append(DetectedDamage(
                        instance_id=idx + 1,
                        damage_type=pred_cls,
                        confidence=pred_conf,
                        bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                        normalized_area=round(norm_area, 4)
                    ))

                st.session_state["confirmed_damages"] = classified_damages
                st.session_state["overrides"] = {}
                st.success(f"✅ Classified {len(classified_damages)} damage region(s) successfully!")
                st.rerun()

    # --- HUMAN-IN-THE-LOOP OVERRIDE PANEL ---
    with col_panel:
        st.markdown("#### 🛠️ AI Predictions & Adjuster Overrides")
        confirmed = st.session_state.get("confirmed_damages", [])
        
        if not confirmed:
            st.info("ℹ️ Draw rectangles on the vehicle photo to mark damage areas, then click **Confirm Damage Selections**.")
        else:
            st.markdown(f"**Identified Damage Instances ({len(confirmed)} total):**")
            
            for dmg in confirmed:
                x1, y1, x2, y2 = [int(v) for v in dmg.bbox_xyxy]
                crop_patch = selected_image.crop((x1, y1, x2, y2))
                
                with st.container():
                    st.markdown(f'<div class="override-card">', unsafe_allow_html=True)
                    c_thumb, c_meta = st.columns([1, 2])
                    
                    with c_thumb:
                        st.image(crop_patch, use_container_width=True)
                    
                    with c_meta:
                        st.markdown(f"**Region #{dmg.instance_id}**")
                        st.markdown(f"• Area: `{dmg.normalized_area*100:.2f}%` of image")
                        st.markdown(f"• AI Model: **{dmg.damage_type.upper()}** (`{dmg.confidence*100:.1f}%` conf)")
                        
                        current_choice = st.session_state.get("overrides", {}).get(dmg.instance_id, dmg.damage_type)
                        override_options = CARDD_CLASSES + ["❌ Delete Region"]
                        
                        choice_idx = CARDD_CLASSES.index(current_choice) if current_choice in CARDD_CLASSES else (
                            len(CARDD_CLASSES) if current_choice == "DELETE" else 0
                        )

                        new_selection = st.selectbox(
                            f"Override Class #{dmg.instance_id}:",
                            options=override_options,
                            index=choice_idx,
                            key=f"override_select_{dmg.instance_id}"
                        )

                        if new_selection != current_choice:
                            if new_selection == "❌ Delete Region":
                                st.session_state["overrides"][dmg.instance_id] = "DELETE"
                            else:
                                st.session_state["overrides"][dmg.instance_id] = new_selection
                            st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)

    # Build Final Active Damage List with Overrides (Setting Confidence = 1.0 on Human Override)
    active_damages: List[DetectedDamage] = []
    for dmg in st.session_state.get("confirmed_damages", []):
        ovr = st.session_state.get("overrides", {}).get(dmg.instance_id, None)
        if ovr == "DELETE":
            continue
        elif ovr is not None and ovr != dmg.damage_type:
            # Human adjuster modified the classification -> set confidence to 1.0 (100% verified)
            active_damages.append(DetectedDamage(
                instance_id=dmg.instance_id,
                damage_type=ovr,
                confidence=1.0,
                bbox_xyxy=dmg.bbox_xyxy,
                normalized_area=dmg.normalized_area
            ))
        else:
            active_damages.append(dmg)

    # If no damages confirmed yet, halt before cost rendering
    if not active_damages:
        st.markdown("---")
        st.info("👆 Please draw boxes and confirm selections above to calculate repair valuation.")
        return

    # -------------------------------------------------------------
    # RUN DOWNSTREAM ACTUARIAL VALUATION & SEVERITY CLASSIFICATION
    # -------------------------------------------------------------
    with st.spinner("Computing Actuarial Repair Cost Breakdown & Monte Carlo Confidence Bounds in INR..."):
        assessment_result = pipeline.assess_image(
            image_input=selected_image,
            vehicle_profile=vehicle_profile,
            user_damages=active_damages,
            filter_background_vehicles=False,
            validate_tires=False,
            validate_glass=False
        )
        active_est = assessment_result.cost_estimate

    st.markdown("---")

    # --- TOP METRIC CARDS (ALL IN INR ₹) ---
    st.markdown(f"### 📊 Claim Valuation Summary ({CURRENCY_SYMBOL})")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    with m1:
        st.metric("Damages Active", f"{len(active_damages)} instance(s)")
    with m2:
        sev_label = assessment_result.predicted_severity.replace('_', ' ').title()
        sev_conf = assessment_result.severity_probabilities.get(assessment_result.predicted_severity, 0.0) * 100
        st.metric("Crash Severity", sev_label, f"{sev_conf:.1f}% conf")
    with m3:
        st.metric("Gross Repair Cost", f"{CURRENCY_SYMBOL}{active_est.estimated_total_cost:,.2f}")
    with m4:
        st.metric("Net Insurer Payout", f"{CURRENCY_SYMBOL}{active_est.net_claim_payout:,.2f}", f"-{CURRENCY_SYMBOL}{deductible:,.0f} Ded.")
    with m5:
        st.metric("90% CI Expected", f"{CURRENCY_SYMBOL}{active_est.cost_p50_median:,.0f}", f"{CURRENCY_SYMBOL}{active_est.cost_p10_optimistic:,.0f} - {CURRENCY_SYMBOL}{active_est.cost_p90_pessimistic:,.0f}")
    with m6:
        st.metric("ML Empirical Benchmark", f"{CURRENCY_SYMBOL}{active_est.ml_empirical_estimate:,.2f}", "26k+ Claims")

    # Constructive Total Loss Status Banner
    if active_est.is_total_loss:
        st.markdown(
            f'<div class="total-loss-banner">⚠️ <b>CONSTRUCTIVE TOTAL LOSS (CTL):</b> Estimated repair cost ({active_est.loss_ratio*100:.1f}% of ACV) exceeds the 75% threshold. Recommend vehicle salvage settlement (Est. Salvage Recovery: {CURRENCY_SYMBOL}{active_est.salvage_value:,.2f}).</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="repairable-banner">✅ <b>REPAIRABLE CLAIM:</b> Estimated repair cost is {active_est.loss_ratio*100:.1f}% of vehicle cash value. Automated repair authorization recommended.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --- MAIN VIEW: Annotated Image & Cost Breakdown ---
    col_img, col_details = st.columns([1.2, 1])

    with col_img:
        st.markdown("#### 🎯 Verified Damage Region Localization")
        annotated_pil = annotate_image_pil(selected_image, active_damages)
        st.image(annotated_pil, caption=f"Confirmed Assessment on {image_name}", use_container_width=True)

    with col_details:
        st.markdown(f"#### 💰 Cost Breakdown by Category ({CURRENCY_SYMBOL})")
        
        # Category Bar Chart in INR
        categories = ["Body Labor", "Paint Labor", "Mech Labor", "OEM Parts", "Paint Mats", "Structural", "Supplies"]
        values = [
            active_est.total_body_labor_cost, active_est.total_paint_labor_cost, active_est.total_mech_labor_cost,
            active_est.total_parts_cost, active_est.total_paint_materials_cost, active_est.structural_overhead_cost,
            active_est.shop_supplies_fee
        ]
        
        fig, ax = plt.subplots(figsize=(7, 4.2))
        colors = ["#2b5c8f", "#3a7bd5", "#43a047", "#e53935", "#fb8c00", "#8e24aa", "#757575"]
        bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(f"Cost in INR ({CURRENCY_SYMBOL})", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.annotate(f"₹{h:,.0f}", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- ITEMIZED DAMAGE TABLE (IN INR) ---
    st.markdown("---")
    st.markdown(f"### 📝 Itemized Damage Ledger & Operations ({CURRENCY_SYMBOL})")
    
    table_rows = []
    for itm in active_est.itemized_damages:
        dmg_match = next((d for d in active_damages if d.instance_id == itm.instance_id), None)
        area_pct = f"{dmg_match.normalized_area * 100:.2f}%" if dmg_match else "N/A"
        conf_display = "100% (Human)" if itm.confidence >= 0.999 else f"{itm.confidence * 100:.1f}%"
        
        table_rows.append({
            "ID": f"#{itm.instance_id}",
            "Damage Type": itm.damage_type.upper(),
            "Confidence": conf_display,
            "Visible Area": area_pct,
            "Action": itm.action,
            "Repair Operation Description": itm.description,
            "Labor Hours (B/P/M)": f"{itm.body_labor_hours} / {itm.paint_labor_hours} / {itm.mech_labor_hours} hrs",
            "Labor Cost": f"{CURRENCY_SYMBOL}{itm.total_labor_cost:,.2f}",
            "OEM Parts Cost": f"{CURRENCY_SYMBOL}{itm.parts_cost:,.2f}",
            "Paint Materials": f"{CURRENCY_SYMBOL}{itm.paint_materials_cost:,.2f}",
            "Item Total": f"{CURRENCY_SYMBOL}{itm.item_subtotal:,.2f}"
        })
    
    df_items = pd.DataFrame(table_rows)
    st.dataframe(df_items, use_container_width=True, hide_index=True)

    # --- CLASSICAL FEATURE INSPECTOR ---
    with st.expander("🔬 Inspect Handcrafted Classical CV Feature Vector (119 Physical Dimensions)", expanded=False):
        st.markdown("Auditable, transparent feature representation extracted purely using classical computer vision (Color moments, GLCM Texture, LBP, Gabor filter banks, 2D FFT, and Reflectance).")
        try:
            from src.features.build_feature_table import extract_all_features_from_image, get_full_feature_dictionary
            sample_feats = extract_all_features_from_image(selected_image)
            dict_df = get_full_feature_dictionary()
            dict_df["Extracted_Value"] = dict_df["Feature_Name"].map(lambda fn: round(sample_feats.get(fn, 0.0), 4))
            
            c_f1, c_f2, c_f3 = st.columns(3)
            c_f1.metric("Color Features", "48 features", "RGB/HSV/k-Means")
            c_f2.metric("Texture Features", "48 features", "GLCM/LBP/Gabor")
            c_f3.metric("Shape/Frequency", "23 features", "Contour/FFT/Glare")

            st.dataframe(dict_df[["Feature_Group", "Feature_Name", "Extracted_Value", "Description"]], use_container_width=True, hide_index=True)
            
            csv_data = dict_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export 119-Dim Feature Vector (.csv)",
                data=csv_data,
                file_name=f"Classical_Features_{vehicle_profile.vehicle_id}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.warning(f"Could not load live feature inspector: {e}")

    # --- FRAUD / ANOMALY AUDIT & EXPORT ---
    st.markdown("---")
    col_audit, col_export = st.columns([1, 1])

    with col_audit:
        st.markdown("#### 🛡️ Claim Fraud & Physical Consistency Audit")
        fa = assessment_result.fraud_audit
        if fa.risk_level == "HIGH":
            st.error(f"Risk Level: HIGH (Anomaly Score: {fa.anomaly_score:.2f}) — Flagged for manual adjuster review!")
        elif fa.risk_level == "MEDIUM":
            st.warning(f"Risk Level: MEDIUM (Anomaly Score: {fa.anomaly_score:.2f}) — Discrepancies detected.")
        else:
            st.success(f"Risk Level: LOW (Anomaly Score: {fa.anomaly_score:.2f}) — Clean Claim Profile.")
            
        for flag in fa.flags:
            st.markdown(f"• {flag}")
            
        if not fa.flags:
            st.markdown("• All computer vision, crash severity, and actuarial cost checks passed consistency bounds.")

    with col_export:
        st.markdown(f"#### 📄 Export Official Claim Adjuster Report ({CURRENCY_SYMBOL})")
        report_text = pipeline.cost_engine.generate_adjuster_text_report(active_est)
        st.download_button(
            label="📥 Download Itemized Claim Report (.txt)",
            data=report_text,
            file_name=f"Claim_Report_{vehicle_profile.vehicle_id}.txt",
            mime="text/plain",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
