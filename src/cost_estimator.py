"""
Knowledge-Grounded Vehicle Repair Cost Estimation Engine.
Integrates YOLOv8 localized damage detections, ResNet50 severity classifications,
and actuarial domain knowledge tables to compute itemized, probabilistic repair costs.
"""

import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from enum import Enum

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    CARDD_CLASSES, SEVERITY_CLASSES, METRICS_DIR, PLOTS_DIR, RANDOM_SEED
)

np.random.seed(RANDOM_SEED)


class VehicleSegment(str, Enum):
    ECONOMY = "economy"          # Hatchbacks, compacts (e.g., Swift, i10, Civic Basic)
    MIDSIZE_SEDAN = "midsize"    # Midsize sedans (e.g., Corolla, Camry, Accord)
    SUV_CROSSOVER = "suv"        # Crossovers, SUVs, Pickup Trucks (e.g., RAV4, CR-V, F-150)
    LUXURY_PREMIUM = "luxury"    # Luxury / German imports (e.g., BMW 3/5, Mercedes C/E, Audi)


# Multipliers per vehicle segment relative to baseline midsize
SEGMENT_MULTIPLIERS = {
    VehicleSegment.ECONOMY: 0.85,
    VehicleSegment.MIDSIZE_SEDAN: 1.00,
    VehicleSegment.SUV_CROSSOVER: 1.25,
    VehicleSegment.LUXURY_PREMIUM: 1.85,
}

# Standard Industry Hourly Labor Rates (USD / hr)
LABOR_RATES = {
    "body": 65.0,        # Standard sheet metal / body labor
    "paint": 70.0,       # Refinish / paint labor
    "mechanical": 85.0,  # Mechanical & electrical labor (sensors, ADAS, lamps, tires)
    "frame": 95.0,       # Frame rack & structural pulling labor
}

# Paint & Material consumables cost per paint labor hour
PAINT_MATERIAL_RATE_PER_HR = 38.0

# Base Severity Multipliers & Structural Surcharges
SEVERITY_FACTORS = {
    "normal": {"multiplier": 1.00, "frame_hours": 0.0, "structural_check": 0.0},
    "moderate_breakage": {"multiplier": 1.40, "frame_hours": 1.5, "structural_check": 150.0},
    "severe_crushed": {"multiplier": 2.25, "frame_hours": 5.0, "structural_check": 450.0},
}

# Actuarial Base Rate Catalogs per Damage Class
DAMAGE_RATE_CATALOG = {
    "dent": {
        "base_body_hours": 2.0,
        "base_paint_hours": 1.8,
        "base_mech_hours": 0.0,
        "base_part_cost": 350.0,    # OEM panel replacement if not repairable
        "replace_area_threshold": 0.08, # >8% bounding area triggers replacement
        "replace_default": False,
        "repair_description": "Panel beating, PDR, filler & surface blending",
        "replace_description": "Body panel replacement & refinish"
    },
    "scratch": {
        "base_body_hours": 0.6,
        "base_paint_hours": 1.5,
        "base_mech_hours": 0.0,
        "base_part_cost": 0.0,
        "replace_area_threshold": 0.20,
        "replace_default": False,
        "repair_description": "Surface sanding, primer coat & clear-coat blend",
        "replace_description": "Full panel repaint & clear-coat"
    },
    "crack": {
        "base_body_hours": 1.8,
        "base_paint_hours": 1.5,
        "base_mech_hours": 0.5,
        "base_part_cost": 380.0,    # Bumper cover or grille
        "replace_area_threshold": 0.05,
        "replace_default": False,
        "repair_description": "Plastic welding, reinforcement & bumper respray",
        "replace_description": "OEM bumper/trim replacement & color match"
    },
    "glass shatter": {
        "base_body_hours": 0.5,
        "base_paint_hours": 0.0,
        "base_mech_hours": 2.5,     # Glass removal, urethane bonding, ADAS calibration
        "base_part_cost": 480.0,    # OEM laminated glass
        "replace_area_threshold": 0.00,
        "replace_default": True,    # Glass is always replaced
        "repair_description": "Resin injection (minor chip only)",
        "replace_description": "OEM windshield/window replacement + ADAS calibration"
    },
    "lamp broken": {
        "base_body_hours": 0.4,
        "base_paint_hours": 0.0,
        "base_mech_hours": 1.2,     # Assembly fitment, wiring harness, aiming
        "base_part_cost": 420.0,    # OEM projector/LED housing
        "replace_area_threshold": 0.00,
        "replace_default": True,    # Broken housing is replaced for safety
        "repair_description": "Lens buffing (minor scuff only)",
        "replace_description": "OEM headlight/taillight assembly & optical aiming"
    },
    "tire flat": {
        "base_body_hours": 0.0,
        "base_paint_hours": 0.0,
        "base_mech_hours": 0.8,     # Mount, balance, TPMS reset
        "base_part_cost": 175.0,    # OEM specification tire
        "replace_area_threshold": 0.00,
        "replace_default": True,    # Flat or damaged tire is replaced
        "repair_description": "Radial puncture patch (tread area only)",
        "replace_description": "New tire replacement, high-speed balancing & alignment"
    }
}


@dataclass
class VehicleProfile:
    """Metadata regarding the insured vehicle under assessment."""
    vehicle_id: str
    make_model: str
    year: int
    segment: VehicleSegment = VehicleSegment.MIDSIZE_SEDAN
    actual_cash_value: float = 22000.0   # Current market value (ACV)
    deductible: float = 500.0             # Policyholder deductible


@dataclass
class DetectedDamage:
    """Structure representing a single damage detection instance from YOLOv8."""
    instance_id: int
    damage_type: str                      # One of CARDD_CLASSES
    confidence: float                     # Detector confidence [0, 1]
    bbox_xyxy: List[float]                # [x1, y1, x2, y2]
    normalized_area: float                # (w * h) / (img_w * img_h) [0, 1]


@dataclass
class ItemizedCostItem:
    """Itemized cost breakdown for a single detected damage instance."""
    instance_id: int
    damage_type: str
    confidence: float
    action: str                           # "REPAIR" or "REPLACE"
    description: str
    body_labor_hours: float
    paint_labor_hours: float
    mech_labor_hours: float
    total_labor_cost: float
    parts_cost: float
    paint_materials_cost: float
    item_subtotal: float


@dataclass
class ClaimCostEstimate:
    """Complete claim assessment cost report."""
    vehicle: VehicleProfile
    predicted_severity: str
    severity_confidence: float
    damage_count: int
    itemized_damages: List[ItemizedCostItem]
    
    # Financial Aggregates
    total_body_labor_cost: float
    total_paint_labor_cost: float
    total_mech_labor_cost: float
    total_labor_cost: float
    total_parts_cost: float
    total_paint_materials_cost: float
    structural_overhead_cost: float
    shop_supplies_fee: float
    subtotal_direct_repair: float
    
    # Final Claim Figures
    estimated_total_cost: float
    policy_deductible: float
    net_claim_payout: float
    
    # Uncertainty Bounds (Monte Carlo 90% Confidence Interval)
    cost_p10_optimistic: float
    cost_p50_median: float
    cost_p90_pessimistic: float
    std_deviation: float
    
    # Constructive Total Loss Analysis
    is_total_loss: bool
    loss_ratio: float                     # Total Cost / ACV
    salvage_value: float


class CostEstimationEngine:
    """
    Actuarial Knowledge-Grounded Repair Cost Estimation Engine.
    Combines computer vision signals with domain pricing matrices and uncertainty modeling.
    """

    def __init__(self, labor_rates: Optional[Dict[str, float]] = None):
        self.labor_rates = labor_rates or LABOR_RATES
        self.damage_catalog = DAMAGE_RATE_CATALOG
        self.severity_factors = SEVERITY_FACTORS
        self.segment_multipliers = SEGMENT_MULTIPLIERS

    def estimate_claim(
        self,
        vehicle: VehicleProfile,
        detected_damages: List[DetectedDamage],
        severity_class: str = "moderate_breakage",
        severity_prob: float = 0.85,
        num_mc_samples: int = 1000
    ) -> ClaimCostEstimate:
        """
        Computes an actuarial repair cost estimate for a vehicle claim.
        """
        sev_info = self.severity_factors.get(severity_class, self.severity_factors["moderate_breakage"])
        sev_mult = sev_info["multiplier"]
        seg_mult = self.segment_multipliers.get(vehicle.segment, 1.0)

        itemized_items: List[ItemizedCostItem] = []
        tot_body_cost = 0.0
        tot_paint_cost = 0.0
        tot_mech_cost = 0.0
        tot_parts_cost = 0.0
        tot_paint_mat_cost = 0.0

        for dmg in detected_damages:
            cat = self.damage_catalog.get(dmg.damage_type, self.damage_catalog["dent"])
            
            # Area Scaling Factor: alpha in [1.0, 3.5]
            area_scale = max(1.0, min(3.5, 1.0 + 0.35 * math.sqrt(dmg.normalized_area * 100.0)))
            
            # Determine Repair vs Replace
            is_replace = cat["replace_default"] or (
                dmg.normalized_area >= cat["replace_area_threshold"] and cat["replace_area_threshold"] > 0
            ) or (severity_class == "severe_crushed" and dmg.damage_type in ["dent", "crack"])

            action_str = "REPLACE" if is_replace else "REPAIR"
            desc = cat["replace_description"] if is_replace else cat["repair_description"]

            # Compute Labor Hours
            if is_replace:
                body_hrs = cat["base_body_hours"] * 1.2
                paint_hrs = cat["base_paint_hours"] * 1.3
                mech_hrs = cat["base_mech_hours"]
                parts_cost = cat["base_part_cost"] * seg_mult
            else:
                body_hrs = cat["base_body_hours"] * area_scale
                paint_hrs = cat["base_paint_hours"] * area_scale
                mech_hrs = cat["base_mech_hours"]
                parts_cost = 0.0

            body_cost = body_hrs * self.labor_rates["body"] * sev_mult * seg_mult
            paint_cost = paint_hrs * self.labor_rates["paint"] * sev_mult * seg_mult
            mech_cost = mech_hrs * self.labor_rates["mechanical"] * sev_mult * seg_mult
            labor_cost = body_cost + paint_cost + mech_cost

            paint_mat_cost = paint_hrs * PAINT_MATERIAL_RATE_PER_HR * seg_mult
            item_subtotal = labor_cost + parts_cost + paint_mat_cost

            tot_body_cost += body_cost
            tot_paint_cost += paint_cost
            tot_mech_cost += mech_cost
            tot_parts_cost += parts_cost
            tot_paint_mat_cost += paint_mat_cost

            itemized_items.append(ItemizedCostItem(
                instance_id=dmg.instance_id,
                damage_type=dmg.damage_type,
                confidence=round(dmg.confidence, 3),
                action=action_str,
                description=desc,
                body_labor_hours=round(body_hrs, 2),
                paint_labor_hours=round(paint_hrs, 2),
                mech_labor_hours=round(mech_hrs, 2),
                total_labor_cost=round(labor_cost, 2),
                parts_cost=round(parts_cost, 2),
                paint_materials_cost=round(paint_mat_cost, 2),
                item_subtotal=round(item_subtotal, 2)
            ))

        tot_labor_cost = tot_body_cost + tot_paint_cost + tot_mech_cost

        # Structural & Overhead Costs
        frame_hrs = sev_info["frame_hours"]
        frame_cost = frame_hrs * self.labor_rates["frame"] * seg_mult
        structural_check = sev_info["structural_check"] * seg_mult
        structural_overhead = frame_cost + structural_check

        # Shop supplies / environmental disposal fee: 6% of labor + paint materials, capped at $150
        shop_supplies = min(150.0, 0.06 * (tot_labor_cost + tot_paint_mat_cost))

        subtotal_direct = (
            tot_labor_cost + tot_parts_cost + tot_paint_mat_cost +
            structural_overhead + shop_supplies
        )

        # Monte Carlo Uncertainty Simulation
        mc_distribution = self._run_monte_carlo(
            base_labor=tot_labor_cost,
            base_parts=tot_parts_cost,
            base_paint_mat=tot_paint_mat_cost,
            overhead=structural_overhead + shop_supplies,
            severity_class=severity_class,
            detected_damages=detected_damages,
            num_samples=num_mc_samples
        )

        p10 = float(np.percentile(mc_distribution, 10))
        p50 = float(np.percentile(mc_distribution, 50))
        p90 = float(np.percentile(mc_distribution, 90))
        std_dev = float(np.std(mc_distribution))

        final_estimated_cost = round(subtotal_direct, 2)
        net_claim = max(0.0, round(final_estimated_cost - vehicle.deductible, 2))

        loss_ratio = final_estimated_cost / vehicle.actual_cash_value if vehicle.actual_cash_value > 0 else 1.0
        is_total_loss = loss_ratio >= 0.75
        salvage_value = round(vehicle.actual_cash_value * 0.22, 2) if is_total_loss else 0.0

        return ClaimCostEstimate(
            vehicle=vehicle,
            predicted_severity=severity_class,
            severity_confidence=round(severity_prob, 3),
            damage_count=len(detected_damages),
            itemized_damages=itemized_items,
            total_body_labor_cost=round(tot_body_cost, 2),
            total_paint_labor_cost=round(tot_paint_cost, 2),
            total_mech_labor_cost=round(tot_mech_cost, 2),
            total_labor_cost=round(tot_labor_cost, 2),
            total_parts_cost=round(tot_parts_cost, 2),
            total_paint_materials_cost=round(tot_paint_mat_cost, 2),
            structural_overhead_cost=round(structural_overhead, 2),
            shop_supplies_fee=round(shop_supplies, 2),
            subtotal_direct_repair=round(subtotal_direct, 2),
            estimated_total_cost=final_estimated_cost,
            policy_deductible=round(vehicle.deductible, 2),
            net_claim_payout=net_claim,
            cost_p10_optimistic=round(p10, 2),
            cost_p50_median=round(p50, 2),
            cost_p90_pessimistic=round(p90, 2),
            std_deviation=round(std_dev, 2),
            is_total_loss=is_total_loss,
            loss_ratio=round(loss_ratio, 3),
            salvage_value=salvage_value
        )

    def _run_monte_carlo(
        self,
        base_labor: float,
        base_parts: float,
        base_paint_mat: float,
        overhead: float,
        severity_class: str,
        detected_damages: List[DetectedDamage],
        num_samples: int = 1000
    ) -> np.ndarray:
        """
        Runs Monte Carlo probabilistic simulation.
        """
        avg_conf = np.mean([d.confidence for d in detected_damages]) if detected_damages else 0.9
        conf_uncertainty_scale = 1.0 + (1.0 - avg_conf) * 0.5

        labor_samples = np.random.normal(
            loc=base_labor,
            scale=base_labor * 0.10 * conf_uncertainty_scale if base_labor > 0 else 1.0,
            size=num_samples
        )

        if base_parts > 0:
            parts_samples = np.random.normal(
                loc=base_parts,
                scale=base_parts * 0.12 * conf_uncertainty_scale,
                size=num_samples
            )
        else:
            parts_samples = np.zeros(num_samples)

        paint_mat_samples = np.random.normal(
            loc=base_paint_mat,
            scale=base_paint_mat * 0.08 if base_paint_mat > 0 else 1.0,
            size=num_samples
        )

        if severity_class == "severe_crushed":
            hidden_damage = np.random.uniform(1.05, 1.35, size=num_samples)
        elif severity_class == "moderate_breakage":
            hidden_damage = np.random.uniform(1.00, 1.18, size=num_samples)
        else:
            hidden_damage = np.random.uniform(1.00, 1.05, size=num_samples)

        total_samples = (labor_samples + parts_samples + paint_mat_samples + overhead) * hidden_damage
        return np.maximum(total_samples, 50.0)

    def generate_adjuster_text_report(self, est: ClaimCostEstimate) -> str:
        """Generates a professional, human-readable actuarial claim assessment document."""
        v = est.vehicle
        lines = []
        lines.append("=" * 80)
        lines.append("          AUTOMATED MOTOR INSURANCE CLAIM ASSESSMENT & ESTIMATE")
        lines.append("=" * 80)
        lines.append(f"Claim Reference ID : CLM-{v.vehicle_id}")
        lines.append(f"Insured Vehicle    : {v.year} {v.make_model} [{v.segment.upper()}]")
        lines.append(f"Actual Cash Value  : ${v.actual_cash_value:,.2f}  |  Policy Deductible: ${v.deductible:,.2f}")
        lines.append(f"Crash Severity     : {est.predicted_severity.upper()} (Confidence: {est.severity_confidence*100:.1f}%)")
        lines.append(f"Damages Detected   : {est.damage_count} distinct localized instance(s)")
        lines.append("-" * 80)
        lines.append(f"{'ID':<4} {'Damage Class':<14} {'Conf':<6} {'Action':<9} {'Labor ($)':<12} {'Parts ($)':<11} {'Total ($)':<10}")
        lines.append("-" * 80)

        for itm in est.itemized_damages:
            lines.append(
                f"{itm.instance_id:<4} {itm.damage_type:<14} {itm.confidence:<6.2f} {itm.action:<9} "
                f"${itm.total_labor_cost:<11.2f} ${itm.parts_cost:<10.2f} ${itm.item_subtotal:<9.2f}"
            )
            lines.append(f"     -> Operation: {itm.description} (Body: {itm.body_labor_hours}h, Paint: {itm.paint_labor_hours}h, Mech: {itm.mech_labor_hours}h)")

        lines.append("-" * 80)
        lines.append("FINANCIAL SUMMARY & COST LEDGER:")
        lines.append(f"  • Total Body Labor Cost       : ${est.total_body_labor_cost:>10,.2f}")
        lines.append(f"  • Total Paint Labor Cost      : ${est.total_paint_labor_cost:>10,.2f}")
        lines.append(f"  • Total Mechanical Labor Cost : ${est.total_mech_labor_cost:>10,.2f}")
        lines.append(f"  • Subtotal Labor Hours Cost   : ${est.total_labor_cost:>10,.2f}")
        lines.append(f"  • Total OEM Replacement Parts : ${est.total_parts_cost:>10,.2f}")
        lines.append(f"  • Paint & Refinish Materials  : ${est.total_paint_materials_cost:>10,.2f}")
        lines.append(f"  • Frame Pulling & Structural  : ${est.structural_overhead_cost:>10,.2f}")
        lines.append(f"  • Shop Supplies & Hazardous   : ${est.shop_supplies_fee:>10,.2f}")
        lines.append("-" * 80)
        lines.append(f"  >>> GROSS ESTIMATED REPAIR COST : ${est.estimated_total_cost:>10,.2f}")
        lines.append(f"  >>> Less Policy Deductible     : -${est.policy_deductible:>9,.2f}")
        lines.append(f"  >>> NET INSURER CLAIM PAYOUT    : ${est.net_claim_payout:>10,.2f}")
        lines.append("-" * 80)
        lines.append("STATISTICAL UNCERTAINTY (Monte Carlo 90% Confidence Bounds):")
        lines.append(f"  • P10 (Optimistic Bound) : ${est.cost_p10_optimistic:>10,.2f}")
        lines.append(f"  • P50 (Expected Median)  : ${est.cost_p50_median:>10,.2f}")
        lines.append(f"  • P90 (Pessimistic Bound): ${est.cost_p90_pessimistic:>10,.2f}")
        lines.append(f"  • Standard Deviation     : ${est.std_deviation:>10,.2f}")
        lines.append("-" * 80)
        lines.append("TOTAL LOSS / SALVAGE DETERMINATION:")
        lines.append(f"  • Loss-to-Value Ratio    : {est.loss_ratio * 100:.1f}% (Threshold: 75.0%)")
        if est.is_total_loss:
            lines.append("  • STATUS                 : [!] CONSTRUCTIVE TOTAL LOSS (CTL)")
            lines.append(f"  • Recommended Action     : Vehicle salvage settlement (Est. Salvage Recovery: ${est.salvage_value:,.2f})")
        else:
            lines.append("  • STATUS                 : [OK] REPAIRABLE")
            lines.append("  • Recommended Action     : Issue automated repair authorization to certified body shop")
        lines.append("=" * 80)
        return "\n".join(lines)


def plot_cost_breakdown(estimate: ClaimCostEstimate, save_path: Path):
    """
    Generates a publication-quality cost breakdown visualization.
    Shows itemized damage costs and cost component proportions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.2, 1]})

    # 1. Cost Categories Bar Chart
    categories = [
        "Body Labor", "Paint Labor", "Mech Labor",
        "OEM Parts", "Paint Mats", "Structural", "Supplies"
    ]
    values = [
        estimate.total_body_labor_cost,
        estimate.total_paint_labor_cost,
        estimate.total_mech_labor_cost,
        estimate.total_parts_cost,
        estimate.total_paint_materials_cost,
        estimate.structural_overhead_cost,
        estimate.shop_supplies_fee
    ]
    colors = ["#2b5c8f", "#3a7bd5", "#43a047", "#e53935", "#fb8c00", "#8e24aa", "#757575"]

    bars = ax1.bar(categories, values, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
    ax1.set_title("Repair Cost Breakdown by Category", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Cost in USD ($)", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=25, ha="right", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax1.annotate(f"${h:,.0f}",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Monte Carlo Uncertainty Distribution
    x_min = max(0.0, estimate.cost_p10_optimistic - 2 * max(estimate.std_deviation, 50.0))
    x_max = estimate.cost_p90_pessimistic + 2 * max(estimate.std_deviation, 50.0)
    x_range = np.linspace(x_min, x_max, 500)
    
    std = max(estimate.std_deviation, 20.0)
    y_dens = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_range - estimate.cost_p50_median) / std) ** 2
    )

    ax2.plot(x_range, y_dens, color="#1e3c72", linewidth=2.5, label="Monte Carlo Cost Density")
    ax2.fill_between(x_range, y_dens, color="#2a5298", alpha=0.25)

    # Add vertical quantile lines
    ax2.axvline(estimate.cost_p10_optimistic, color="#43a047", linestyle="--", linewidth=2,
                label=f"P10 (Optimistic): ${estimate.cost_p10_optimistic:,.0f}")
    ax2.axvline(estimate.cost_p50_median, color="#e65100", linestyle="-", linewidth=2.2,
                label=f"P50 (Median): ${estimate.cost_p50_median:,.0f}")
    ax2.axvline(estimate.cost_p90_pessimistic, color="#c62828", linestyle="--", linewidth=2,
                label=f"P90 (Pessimistic): ${estimate.cost_p90_pessimistic:,.0f}")

    ax2.set_title(f"Uncertainty Bounds: ${estimate.estimated_total_cost:,.2f} Expected",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Estimated Total Repair Cost ($)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(
        f"Automated Claim Valuation: {estimate.vehicle.make_model} ({estimate.predicted_severity.upper()})\n"
        f"Net Insurer Payout: USD {estimate.net_claim_payout:,.2f} (Deductible: USD {estimate.policy_deductible:,.0f})",
        fontsize=14, fontweight="bold", y=1.03
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Cost Breakdown Visualization to: {save_path}", flush=True)


def benchmark_cost_engine_across_scenarios():
    """
    Executes a comprehensive validation benchmark across diverse insurance claim scenarios.
    """
    print("\n" + "="*70)
    print("EXECUTING PHASE 7: KNOWLEDGE-GROUNDED REPAIR COST ENGINE BENCHMARK")
    print("="*70, flush=True)

    engine = CostEstimationEngine()

    scenarios = [
        {
            "name": "Scenario 1: Minor Parking Lot Scraping",
            "vehicle": VehicleProfile(
                vehicle_id="V-2024-001",
                make_model="Toyota Corolla LE",
                year=2021,
                segment=VehicleSegment.MIDSIZE_SEDAN,
                actual_cash_value=18500.0,
                deductible=500.0
            ),
            "damages": [
                DetectedDamage(1, "scratch", 0.92, [100, 200, 250, 230], 0.012),
                DetectedDamage(2, "dent", 0.88, [150, 210, 320, 310], 0.025),
            ],
            "severity": "normal",
            "severity_prob": 0.94
        },
        {
            "name": "Scenario 2: Moderate Front-End Collision",
            "vehicle": VehicleProfile(
                vehicle_id="V-2024-002",
                make_model="Honda CR-V EX",
                year=2022,
                segment=VehicleSegment.SUV_CROSSOVER,
                actual_cash_value=27000.0,
                deductible=500.0
            ),
            "damages": [
                DetectedDamage(1, "dent", 0.91, [120, 150, 400, 380], 0.065),
                DetectedDamage(2, "crack", 0.87, [180, 320, 410, 420], 0.038),
                DetectedDamage(3, "lamp broken", 0.95, [380, 140, 520, 280], 0.028),
            ],
            "severity": "moderate_breakage",
            "severity_prob": 0.89
        },
        {
            "name": "Scenario 3: Severe High-Speed T-Bone Crash",
            "vehicle": VehicleProfile(
                vehicle_id="V-2024-003",
                make_model="BMW 330i xDrive",
                year=2020,
                segment=VehicleSegment.LUXURY_PREMIUM,
                actual_cash_value=32000.0,
                deductible=1000.0
            ),
            "damages": [
                DetectedDamage(1, "dent", 0.94, [50, 80, 580, 500], 0.220),
                DetectedDamage(2, "crack", 0.89, [120, 300, 450, 520], 0.090),
                DetectedDamage(3, "glass shatter", 0.96, [150, 50, 500, 280], 0.085),
                DetectedDamage(4, "lamp broken", 0.93, [480, 220, 600, 360], 0.035),
                DetectedDamage(5, "tire flat", 0.90, [80, 380, 240, 540], 0.045),
            ],
            "severity": "severe_crushed",
            "severity_prob": 0.96
        },
        {
            "name": "Scenario 4: Economy Total-Loss Claim",
            "vehicle": VehicleProfile(
                vehicle_id="V-2024-004",
                make_model="Maruti Suzuki Swift",
                year=2016,
                segment=VehicleSegment.ECONOMY,
                actual_cash_value=4800.0,
                deductible=250.0
            ),
            "damages": [
                DetectedDamage(1, "dent", 0.95, [40, 60, 550, 480], 0.210),
                DetectedDamage(2, "lamp broken", 0.92, [420, 180, 580, 320], 0.030),
                DetectedDamage(3, "crack", 0.88, [100, 280, 480, 490], 0.080),
                DetectedDamage(4, "glass shatter", 0.94, [120, 40, 460, 260], 0.075),
            ],
            "severity": "severe_crushed",
            "severity_prob": 0.95
        }
    ]

    benchmark_outputs = []

    for sc in scenarios:
        print(f"\nEvaluating: {sc['name']}...")
        estimate = engine.estimate_claim(
            vehicle=sc["vehicle"],
            detected_damages=sc["damages"],
            severity_class=sc["severity"],
            severity_prob=sc["severity_prob"]
        )

        report_txt = engine.generate_adjuster_text_report(estimate)
        print(report_txt)

        if sc["vehicle"].vehicle_id == "V-2024-002":
            plot_path = PLOTS_DIR / "cost_breakdown_scenario2.png"
            plot_cost_breakdown(estimate, plot_path)

        benchmark_outputs.append({
            "scenario": sc["name"],
            "vehicle_id": sc["vehicle"].vehicle_id,
            "make_model": sc["vehicle"].make_model,
            "segment": sc["vehicle"].segment.value,
            "actual_cash_value": sc["vehicle"].actual_cash_value,
            "severity": sc["severity"],
            "damage_count": estimate.damage_count,
            "total_labor_cost": estimate.total_labor_cost,
            "total_parts_cost": estimate.total_parts_cost,
            "total_paint_materials": estimate.total_paint_materials_cost,
            "structural_overhead": estimate.structural_overhead_cost,
            "gross_repair_cost": estimate.estimated_total_cost,
            "deductible": estimate.policy_deductible,
            "net_payout": estimate.net_claim_payout,
            "p10_optimistic": estimate.cost_p10_optimistic,
            "p50_median": estimate.cost_p50_median,
            "p90_pessimistic": estimate.cost_p90_pessimistic,
            "is_total_loss": estimate.is_total_loss,
            "loss_ratio": estimate.loss_ratio,
            "salvage_value": estimate.salvage_value,
            "itemized_ledger": [asdict(itm) for itm in estimate.itemized_damages]
        })

    summary_file = METRICS_DIR / "phase7_cost_estimation_benchmark.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 7 — Knowledge-Grounded Repair Cost Estimation Engine",
            "rate_tables": {
                "labor_rates_usd_hr": LABOR_RATES,
                "paint_materials_usd_hr": PAINT_MATERIAL_RATE_PER_HR,
                "segment_multipliers": {k.value: v for k, v in SEGMENT_MULTIPLIERS.items()},
                "severity_multipliers": SEVERITY_FACTORS
            },
            "benchmark_scenarios": benchmark_outputs
        }, f, indent=2)

    print(f"\n[SUCCESS] Exported Phase 7 Cost Estimation Benchmark to: {summary_file}")
    return benchmark_outputs


if __name__ == "__main__":
    benchmark_cost_engine_across_scenarios()
