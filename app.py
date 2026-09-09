"""
Interactive AI Vehicle Damage Assessment, Inspection & Actuarial Repair Cost Estimator.
Streamlit Web Dashboard for Automated Insurance Claim Adjudication.
"""

import sys
import os
import time
from pathlib import Path

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from src.config import CARDD_CLASSES, SEVERITY_CLASSES, PROCESSED_DATA_DIR
from src.cost_estimator import VehicleProfile, VehicleSegment, ClaimCostEstimate
from src.pipeline import VehicleDamageAssessmentPipeline, CLASS_COLORS, FullAssessmentResult

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="AI Vehicle Damage & Repair Cost Estimator",
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
        padding: 12px 18px;
        border-radius: 6px;
        color: #ff6b6b;
        font-weight: 600;
        margin: 15px 0;
    }
    .repairable-banner {
        background: linear-gradient(90deg, rgba(46,204,113,0.2) 0%, rgba(39,174,96,0.1) 100%);
        border-left: 5px solid #2ecc71;
        padding: 12px 18px;
        border-radius: 6px;
        color: #2ecc71;
        font-weight: 600;
        margin: 15px 0;
    }
    .info-filter-banner {
        background: linear-gradient(90deg, rgba(52,152,219,0.15) 0%, rgba(41,128,185,0.05) 100%);
        border-left: 4px solid #3498db;
        padding: 8px 14px;
        border-radius: 6px;
        color: #5dade2;
        font-size: 13px;
        margin-bottom: 12px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI Vision & Cost Estimation Models...")
def get_pipeline():
    """Initializes and caches the end-to-end assessment pipeline."""
    return VehicleDamageAssessmentPipeline()


def annotate_image_pil(
    image: Image.Image,
    damages: list,
    primary_roi: list = None
) -> Image.Image:
    """Draws color-coded bounding boxes and label badges onto a PIL Image."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # Draw Primary Subject Vehicle RoI if present
    if primary_roi is not None:
        vx1, vy1, vx2, vy2 = primary_roi
        draw.rectangle([vx1, vy1, vx2, vy2], outline=(255, 215, 0), width=3)
        draw.rectangle([vx1, max(0, vy1 - 22), vx1 + 220, vy1], fill=(255, 215, 0))
        draw.text((vx1 + 6, max(0, vy1 - 19)), "PRIMARY CLAIM VEHICLE", fill=(0, 0, 0))

    for dmg in damages:
        x1, y1, x2, y2 = dmg.bbox_xyxy
        color = CLASS_COLORS.get(dmg.damage_type, (0, 255, 0))
        
        # Bounding Box outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Label Badge Header
        label = f"#{dmg.instance_id} {dmg.damage_type.upper()} ({dmg.confidence*100:.0f}%)"
        text_bbox = draw.textbbox((x1, y1), label)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        
        badge_y0 = max(0, y1 - th - 6)
        draw.rectangle([x1, badge_y0, x1 + tw + 10, y1], fill=color)
        draw.text((x1 + 5, badge_y0 + 2), label, fill=(255, 255, 255))
        
    return annotated


def main():
    st.title("🚗 AI Vehicle Damage & Repair Cost Estimator")
    st.markdown("Automated Computer Vision Damage Localization, Crash Severity Classification & Empirical Actuarial Cost Assessment")

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

    acv = st.sidebar.number_input("Actual Cash Value (ACV $)", min_value=1000.0, max_value=150000.0, value=50000.0, step=500.0)
    deductible = st.sidebar.number_input("Policy Deductible ($)", min_value=0.0, max_value=5000.0, value=500.0, step=100.0)

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ AI Detector & Intelligent Filter Settings")
    
    filter_bg = st.sidebar.checkbox(
        "🎯 Filter Background Vehicles (Focus on Subject Car)",
        value=True,
        help="Eliminates false-positive detections on background vehicles, parked cars, and workshop scenery."
    )
    
    validate_tires_toggle = st.sidebar.checkbox(
        "🛞 Validate Tire Deflation (Suppress Normal Wheel False Positives)",
        value=True,
        help="Analyzes geometric wheel circularity and contact-patch deflation to verify genuine flat tires vs normal inflated alloy wheels."
    )

    conf_thresh = st.sidebar.slider(
        "Base Damage Confidence Threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.20,
        step=0.05
    )

    with st.sidebar.expander("🛠️ Advanced Per-Class Confidence Calibration"):
        dent_conf = st.slider("Dent Min Conf", 0.10, 0.80, max(0.15, conf_thresh), 0.05)
        scratch_conf = st.slider("Scratch Min Conf", 0.10, 0.80, max(0.15, conf_thresh), 0.05)
        crack_conf = st.slider("Crack Min Conf", 0.10, 0.80, max(0.20, conf_thresh), 0.05)
        glass_conf = st.slider("Glass Shatter Min Conf", 0.10, 0.80, max(0.25, conf_thresh), 0.05)
        lamp_conf = st.slider("Lamp Broken Min Conf", 0.10, 0.80, max(0.25, conf_thresh), 0.05)
        tire_conf = st.slider("Tire Flat Min Conf", 0.30, 0.90, 0.50, 0.05)

    custom_class_confs = {
        "dent": dent_conf,
        "scratch": scratch_conf,
        "crack": crack_conf,
        "glass shatter": glass_conf,
        "lamp broken": lamp_conf,
        "tire flat": tire_conf
    }

    # Preset Sample Images from Test Split
    test_img_dir = PROCESSED_DATA_DIR / "cardd" / "images" / "test"
    sample_files = []
    if test_img_dir.exists():
        sample_files = sorted([f.name for f in test_img_dir.glob("*.jpg")])[:15]

    st.sidebar.markdown("---")
    st.sidebar.header("🖼️ Test Dataset Presets")
    sample_select = st.sidebar.selectbox(
        "Or pick a sample from held-out test set:",
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
        st.info("👆 Please upload a vehicle photo or select a sample image from the left sidebar to run the assessment.")
        return

    # Build Vehicle Profile
    vehicle_profile = VehicleProfile(
        vehicle_id="CLM-" + str(int(time.time()))[-6:],
        make_model=make_model,
        year=year,
        segment=selected_segment,
        actual_cash_value=acv,
        deductible=deductible
    )

    # Run Multi-Stage Pipeline
    with st.spinner("Running Intelligent Vehicle RoI Localization, Damage Detection & Cost Estimation..."):
        result = pipeline.assess_image(
            image_input=selected_image,
            vehicle_profile=vehicle_profile,
            conf_threshold=conf_thresh,
            filter_background_vehicles=filter_bg,
            validate_tires=validate_tires_toggle,
            min_conf_per_class=custom_class_confs
        )

    # Display Filter Notification Banner if items were filtered
    if result.filtered_background_damages_count > 0 or result.suppressed_tire_false_positives_count > 0:
        filter_notices = []
        if result.filtered_background_damages_count > 0:
            filter_notices.append(f"🎯 Filtered **{result.filtered_background_damages_count}** background vehicle detection(s)")
        if result.suppressed_tire_false_positives_count > 0:
            filter_notices.append(f"🛞 Suppressed **{result.suppressed_tire_false_positives_count}** normal inflated wheel false-positive(s)")
        
        st.markdown(
            f'<div class="info-filter-banner">ℹ️ <b>Intelligent Noise Suppression Active:</b> {" | ".join(filter_notices)}</div>',
            unsafe_allow_html=True
        )

    # --- TOP METRIC CARDS ---
    st.markdown("### 📊 Claim Valuation Summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    with m1:
        st.metric("Damages Detected", f"{len(result.detected_damages)} instance(s)")
    with m2:
        sev_label = result.predicted_severity.replace('_', ' ').title()
        sev_conf = result.severity_probabilities.get(result.predicted_severity, 0.0) * 100
        st.metric("Crash Severity", sev_label, f"{sev_conf:.1f}% conf")
    with m3:
        st.metric("Gross Repair Cost", f"${result.cost_estimate.estimated_total_cost:,.2f}")
    with m4:
        st.metric("Net Insurer Payout", f"${result.cost_estimate.net_claim_payout:,.2f}", f"-${deductible:,.0f} Ded.")
    with m5:
        st.metric("90% CI Expected", f"${result.cost_estimate.cost_p50_median:,.0f}", f"${result.cost_estimate.cost_p10_optimistic:,.0f} - ${result.cost_estimate.cost_p90_pessimistic:,.0f}")
    with m6:
        st.metric("ML Empirical Benchmark", f"${result.cost_estimate.ml_empirical_estimate:,.2f}", "26k+ Claims")

    # Total Loss Status Banner
    if result.cost_estimate.is_total_loss:
        st.markdown(
            f'<div class="total-loss-banner">⚠️ CONSTRUCTIVE TOTAL LOSS: Estimated repair cost ({result.cost_estimate.loss_ratio*100:.1f}% of ACV) exceeds the 75% threshold. Recommend salvage recovery (Est. Salvage: ${result.cost_estimate.salvage_value:,.2f}).</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="repairable-banner">✅ REPAIRABLE CLAIM: Estimated repair cost is {result.cost_estimate.loss_ratio*100:.1f}% of vehicle cash value. Automated repair authorization recommended.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --- MAIN VIEW: Annotated Image & Cost Breakdown ---
    col_img, col_details = st.columns([1.2, 1])

    with col_img:
        st.markdown("#### 🎯 AI Damage Localization")
        annotated_pil = annotate_image_pil(selected_image, result.detected_damages, primary_roi=result.primary_vehicle_roi)
        st.image(annotated_pil, caption=f"Assessment on {image_name} (Inference: {result.processing_time_ms:.0f} ms)", use_container_width=True)

    with col_details:
        st.markdown("#### 💰 Cost Breakdown by Category")
        
        # Category Bar Chart
        est = result.cost_estimate
        categories = ["Body Labor", "Paint Labor", "Mech Labor", "OEM Parts", "Paint Mats", "Structural", "Supplies"]
        values = [
            est.total_body_labor_cost, est.total_paint_labor_cost, est.total_mech_labor_cost,
            est.total_parts_cost, est.total_paint_materials_cost, est.structural_overhead_cost,
            est.shop_supplies_fee
        ]
        
        fig, ax = plt.subplots(figsize=(7, 4.2))
        colors = ["#2b5c8f", "#3a7bd5", "#43a047", "#e53935", "#fb8c00", "#8e24aa", "#757575"]
        bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel("Cost ($)", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.annotate(f"${h:,.0f}", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- ITEMIZED DAMAGE TABLE ---
    st.markdown("### 📝 Itemized Damage Ledger & Operations")
    if result.detected_damages:
        table_rows = []
        for itm in est.itemized_damages:
            dmg_match = next((d for d in result.detected_damages if d.instance_id == itm.instance_id), None)
            area_pct = f"{dmg_match.normalized_area * 100:.2f}%" if dmg_match else "N/A"
            
            table_rows.append({
                "ID": f"#{itm.instance_id}",
                "Damage Type": itm.damage_type.upper(),
                "Confidence": f"{itm.confidence * 100:.1f}%",
                "Visible Area": area_pct,
                "Action": itm.action,
                "Repair Operation Description": itm.description,
                "Labor Hours (B/P/M)": f"{itm.body_labor_hours} / {itm.paint_labor_hours} / {itm.mech_labor_hours} hrs",
                "Labor Cost": f"${itm.total_labor_cost:,.2f}",
                "OEM Parts Cost": f"${itm.parts_cost:,.2f}",
                "Paint Materials": f"${itm.paint_materials_cost:,.2f}",
                "Item Total": f"${itm.item_subtotal:,.2f}"
            })
        
        df_items = pd.DataFrame(table_rows)
        st.dataframe(df_items, use_container_width=True, hide_index=True)
    else:
        st.info("No localized damage instances detected on the primary subject vehicle above current threshold.")

    # --- FRAUD / ANOMALY AUDIT & EXPORT ---
    st.markdown("---")
    col_audit, col_export = st.columns([1, 1])

    with col_audit:
        st.markdown("#### 🛡️ Claim Fraud & Anomaly Audit")
        fa = result.fraud_audit
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
        st.markdown("#### 📄 Export Official Claim Adjuster Report")
        report_text = pipeline.cost_engine.generate_adjuster_text_report(result.cost_estimate)
        st.download_button(
            label="📥 Download Itemized Claim Report (.txt)",
            data=report_text,
            file_name=f"Claim_Report_{vehicle_profile.vehicle_id}.txt",
            mime="text/plain",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
