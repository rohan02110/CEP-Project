"""
Interactive Web Dashboard for Vehicle Damage Assessment & Claim Valuation.
Streamlit Application for real-time AI damage localization, severity classification,
fraud audit, and actuarial repair cost estimation.
"""

import os
import sys
import io
import time
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import CARDD_CLASSES, SEVERITY_CLASSES, PROCESSED_DATA_DIR
from src.cost_estimator import VehicleProfile, VehicleSegment, CostEstimationEngine
from src.pipeline import VehicleDamageAssessmentPipeline, CLASS_COLORS


# Page Configuration
st.set_page_config(
    page_title="AI Vehicle Damage & Repair Cost Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean Dashboard UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .total-loss-banner {
        background-color: #ffebee;
        border-left: 6px solid #d32f2f;
        padding: 12px;
        border-radius: 6px;
        color: #b71c1c;
        font-weight: bold;
        margin: 10px 0;
    }
    .repairable-banner {
        background-color: #e8f5e9;
        border-left: 6px solid #388e3c;
        padding: 12px;
        border-radius: 6px;
        color: #1b5e20;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI Models (YOLOv8 + ResNet50 + Severity Classifier)...")
def load_pipeline():
    """Initializes and caches the multi-stage assessment pipeline."""
    return VehicleDamageAssessmentPipeline()


def annotate_image_pil(pil_img: Image.Image, detected_damages):
    """Draws color-coded bounding boxes and labels on PIL image."""
    import cv2
    img_np = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = img_np.shape

    for dmg in detected_damages:
        x1, y1, x2, y2 = [int(v) for v in dmg.bbox_xyxy]
        color = CLASS_COLORS.get(dmg.damage_type, (0, 255, 0))
        bgr_color = (color[2], color[1], color[0])

        # Draw Bounding Box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), bgr_color, 3)

        # Label tag background
        label_text = f"#{dmg.instance_id} {dmg.damage_type.upper()} ({dmg.confidence*100:.0f}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_np, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), bgr_color, -1)
        cv2.putText(
            img_np, label_text, (x1 + 5, max(15, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

    return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))


def main():
    st.markdown('<div class="main-header">🚗 AI Vehicle Damage & Repair Cost Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Computer Vision Damage Localization, Crash Severity Classification & Actuarial Cost Assessment</div>', unsafe_allow_html=True)

    pipeline = load_pipeline()

    # --- SIDEBAR: Claim & Vehicle Parameters ---
    st.sidebar.header("📋 Vehicle & Policy Details")
    
    make_model = st.sidebar.text_input("Vehicle Make & Model", value="Honda Civic EX")
    year = st.sidebar.number_input("Model Year", min_value=2000, max_value=2026, value=2021)
    
    segment_choice = st.sidebar.selectbox(
        "Vehicle Category",
        options=["Economy / Compact", "Midsize Sedan", "SUV / Crossover", "Luxury / Premium"],
        index=1
    )
    segment_map = {
        "Economy / Compact": VehicleSegment.ECONOMY,
        "Midsize Sedan": VehicleSegment.MIDSIZE_SEDAN,
        "SUV / Crossover": VehicleSegment.SUV_CROSSOVER,
        "Luxury / Premium": VehicleSegment.LUXURY_PREMIUM
    }
    selected_segment = segment_map[segment_choice]

    acv = st.sidebar.number_input("Actual Cash Value (ACV $)", min_value=1000.0, max_value=150000.0, value=22000.0, step=500.0)
    deductible = st.sidebar.number_input("Policy Deductible ($)", min_value=0.0, max_value=5000.0, value=500.0, step=100.0)

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ AI Detector Settings")
    conf_thresh = st.sidebar.slider("YOLOv8 Damage Confidence Threshold", min_value=0.10, max_value=0.80, value=0.20, step=0.05)

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
    with st.spinner("Running Damage Detection, Severity Classification & Actuarial Valuation..."):
        result = pipeline.assess_image(
            image_input=selected_image,
            vehicle_profile=vehicle_profile,
            conf_threshold=conf_thresh
        )

    # --- TOP METRIC CARDS ---
    st.markdown("### 📊 Claim Valuation Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    
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
        annotated_pil = annotate_image_pil(selected_image, result.detected_damages)
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
            # find corresponding detected damage for area
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
        st.info("No localized damage instances detected above the current confidence threshold.")

    # --- FRAUD / ANOMALY AUDIT & EXPORT ---
    st.markdown("---")
    col_audit, col_export = st.columns([1, 1])

    with col_audit:
        st.markdown("#### 🛡️ Claim Fraud & Anomaly Audit")
        f_audit = result.fraud_audit
        if f_audit.risk_level == "LOW":
            st.success(f"Risk Level: **LOW** (Anomaly Score: {f_audit.anomaly_score:.2f}) — Claim heuristics consistent with visual damage.")
        elif f_audit.risk_level == "MEDIUM":
            st.warning(f"Risk Level: **MEDIUM** (Anomaly Score: {f_audit.anomaly_score:.2f})")
        else:
            st.error(f"Risk Level: **HIGH** (Anomaly Score: {f_audit.anomaly_score:.2f}) — Flagged for manual adjuster review!")

        if f_audit.flags:
            for flag in f_audit.flags:
                st.write(f"- {flag}")

    with col_export:
        st.markdown("#### 📄 Export Official Claim Adjuster Report")
        report_text = pipeline.cost_engine.generate_adjuster_text_report(result.cost_estimate)
        st.download_button(
            label="📥 Download Itemized Claim Report (.txt)",
            data=report_text,
            file_name=f"Claim_Report_{vehicle_profile.vehicle_id}_{image_name.split('.')[0]}.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
