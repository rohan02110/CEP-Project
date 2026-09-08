"""
CLI Interactive Tool for Testing the AI Motor Insurance Claim Pipeline on Any Image.

Usage Examples:
    # Test a specific image from the dataset
    python src/test_claim.py --image data/processed/cardd/images/test/000046.jpg

    # Test with custom vehicle parameters
    python src/test_claim.py --image path/to/my_car.jpg --model "Toyota Camry" --year 2022 --segment midsize --acv 22000 --deductible 500

    # Test a random sample from the test set
    python src/test_claim.py --random
"""

import os
import sys
import random
import argparse
from pathlib import Path

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import PROCESSED_DATA_DIR, PLOTS_DIR
from src.cost_estimator import VehicleProfile, VehicleSegment
from src.pipeline import VehicleDamageAssessmentPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Test AI Motor Insurance Assessment Pipeline on an image.")
    parser.add_argument("--image", type=str, default=None, help="Path to vehicle damage image.")
    parser.add_argument("--random", action="store_true", help="Pick a random test image from the dataset.")
    parser.add_argument("--make", type=str, default="Honda Civic", help="Vehicle make and model name.")
    parser.add_argument("--year", type=int, default=2021, help="Model year of vehicle.")
    parser.add_argument("--segment", type=str, choices=["economy", "midsize", "suv", "luxury"], default="midsize", help="Vehicle category.")
    parser.add_argument("--acv", type=float, default=22000.0, help="Actual Cash Value (market value) in USD.")
    parser.add_argument("--deductible", type=float, default=500.0, help="Policyholder deductible in USD.")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLOv8 damage confidence threshold.")
    parser.add_argument("--out", type=str, default=None, help="Output plot destination path.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine image path
    if args.random or args.image is None:
        test_dir = PROCESSED_DATA_DIR / "cardd" / "images" / "test"
        all_test_imgs = list(test_dir.glob("*.jpg"))
        if not all_test_imgs:
            print(f"No test images found in: {test_dir}")
            sys.exit(1)
        image_path = random.choice(all_test_imgs)
        print(f"Selected Random Test Image: {image_path.name}")
    else:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found at {image_path}")
            sys.exit(1)

    # Build Vehicle Profile
    segment_enum = VehicleSegment(args.segment)
    vehicle = VehicleProfile(
        vehicle_id="CLI-TEST-" + image_path.stem,
        make_model=args.make,
        year=args.year,
        segment=segment_enum,
        actual_cash_value=args.acv,
        deductible=args.deductible
    )

    out_plot = Path(args.out) if args.out else PLOTS_DIR / f"claim_assessment_{image_path.stem}.png"

    # Run Pipeline
    pipeline = VehicleDamageAssessmentPipeline()
    result = pipeline.assess_image(
        image_input=image_path,
        vehicle_profile=vehicle,
        conf_threshold=args.conf,
        output_plot_path=out_plot
    )

    # Print Adjuster Text Report
    report = pipeline.cost_engine.generate_adjuster_text_report(result.cost_estimate)
    print("\n" + report)

    print("\n" + "="*80)
    print(f"[PIPELINE SUMMARY]")
    print(f"• Image Path          : {image_path}")
    print(f"• Damages Detected    : {len(result.detected_damages)}")
    for d in result.detected_damages:
        print(f"   - #{d.instance_id} {d.damage_type.upper()} ({d.confidence*100:.1f}% conf, area: {d.normalized_area*100:.2f}%)")
    print(f"• Crash Severity      : {result.predicted_severity.upper()} ({result.severity_probabilities})")
    print(f"• Gross Repair Cost   : USD {result.cost_estimate.estimated_total_cost:,.2f}")
    print(f"• Net Claim Payout    : USD {result.cost_estimate.net_claim_payout:,.2f}")
    print(f"• 90% Confidence      : USD {result.cost_estimate.cost_p10_optimistic:,.2f} - USD {result.cost_estimate.cost_p90_pessimistic:,.2f}")
    print(f"• Total Loss Status   : {'[!] CONSTRUCTIVE TOTAL LOSS' if result.cost_estimate.is_total_loss else '[OK] REPAIRABLE'}")
    print(f"• Fraud / Anomaly     : {result.fraud_audit.risk_level} (Score: {result.fraud_audit.anomaly_score:.2f})")
    print(f"• Latency             : {result.processing_time_ms:.1f} ms")
    print(f"• Visual Report Saved : {out_plot}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
