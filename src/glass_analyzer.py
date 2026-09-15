"""
Advanced Multi-Cue Glass & Window Integrity Diagnostic Analyzer.
Analyzes micro-fracture networks, edge orientation entropy, Laplacian high-frequency variance,
and specular reflection masks to accurately distinguish shattered/cracked vehicle glass from intact windows.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import numpy as np
from PIL import Image


@dataclass
class GlassDiagnosticResult:
    """Diagnostic breakdown for window / glass integrity analysis."""
    status: str                        # "SHATTERED", "CRACKED", "INTACT"
    is_shattered_or_cracked: bool      # True if damage is confirmed
    shatter_confidence: float          # Composite confidence score [0.0, 1.0]
    fracture_density: float            # Ratio of micro-fracture edges in non-glare area [0.0, 1.0]
    gradient_entropy: float            # Directional dispersion entropy of edges [0.0, 1.0]
    laplacian_variance: float          # High-frequency texture sharpness / dicing variance
    glare_ratio: float                 # Percentage of crop covered by smooth specular glare [0.0, 1.0]
    reason: str                        # Explanatory verdict description
    fracture_heatmap_rgb: Optional[np.ndarray] = None  # Colorized fracture intensity overlay


class GlassIntegrityAnalyzer:
    """
    Multi-Cue Computer Vision & Texture Diagnostic Engine for Vehicle Glass:
    1. Micro-Fracture Spiderweb Edge Density (multi-scale Canny & Sobel gradients)
    2. Gradient Orientation Entropy (omnidirectional shatter networks vs unidirectional horizon/glare reflections)
    3. Laplacian Texture Sharpness & Tempered Glass Dicing Analysis
    4. Specular Glare & Sun Reflection Masking & Disambiguation
    """

    def __init__(
        self,
        min_crop_size: int = 16,
        fracture_density_thresh: float = 0.045,
        entropy_thresh: float = 0.55,
        laplacian_thresh: float = 85.0
    ):
        self.min_crop_size = min_crop_size
        self.fracture_density_thresh = fracture_density_thresh
        self.entropy_thresh = entropy_thresh
        self.laplacian_thresh = laplacian_thresh

    def compute_gradient_entropy(self, gray: np.ndarray, edge_mask: np.ndarray) -> float:
        """
        Calculates the Shannon entropy of edge gradient orientations.
        - Unidirectional reflections (e.g. horizon line, building edge, glare streaks) have low entropy.
        - Shattered glass with complex spiderweb/mosaic fragmentation has high isotropic entropy.
        Returns normalized entropy in [0.0, 1.0].
        """
        if np.count_nonzero(edge_mask) < 20:
            return 0.0

        # Compute Sobel gradients in X and Y
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate angles in degrees [0, 180)
        angles = np.arctan2(sobel_y, sobel_x) * (180.0 / np.pi)
        angles = np.mod(angles, 180.0)

        # Sample angles only at edge locations
        edge_angles = angles[edge_mask > 0]
        if len(edge_angles) < 20:
            return 0.0

        # Histogram of orientations into 18 bins (10 deg each)
        hist, _ = np.histogram(edge_angles, bins=18, range=(0, 180), density=True)
        hist = hist[hist > 0]

        # Shannon Entropy
        entropy = -np.sum(hist * np.log2(hist))
        # Max entropy for 18 uniform bins is log2(18) = ~4.17
        max_entropy = np.log2(18.0)
        norm_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
        return norm_entropy

    def detect_specular_glare(self, bgr_crop: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detects smooth specular highlights and sun glare regions that could cause false positives.
        Returns: (glare_binary_mask, glare_area_ratio)
        """
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        # Specular glare typically has very high brightness (V > 215) and low saturation (S < 65)
        glare_mask = (v_channel > 215) & (s_channel < 65)
        glare_mask_u8 = glare_mask.astype(np.uint8) * 255

        # Morphological opening to keep significant glare blobs, discard isolated tiny noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask_cleaned = cv2.morphologyEx(glare_mask_u8, cv2.MORPH_OPEN, kernel)

        total_pixels = bgr_crop.shape[0] * bgr_crop.shape[1]
        glare_ratio = float(np.count_nonzero(glare_mask_cleaned) / max(1, total_pixels))
        return glare_mask_cleaned, glare_ratio

    def compute_fracture_features(
        self,
        bgr_crop: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extracts multi-scale fracture features from a window / glass image crop.
        """
        h, w = bgr_crop.shape[:2]
        if h < self.min_crop_size or w < self.min_crop_size:
            return {
                "fracture_density": 0.0,
                "gradient_entropy": 0.0,
                "laplacian_variance": 0.0,
                "glare_ratio": 0.0,
                "fracture_mask": np.zeros((max(1, h), max(1, w)), dtype=np.uint8),
                "crack_branch_count": 0
            }

        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        
        # Specular glare detection
        glare_mask, glare_ratio = self.detect_specular_glare(bgr_crop)

        # High-frequency texture sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian_var = float(laplacian.var())

        # Bilateral filter to preserve sharp crack edges while smoothing uniform glass reflection noise
        filtered_gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # Multi-scale edge detection for hairline to heavy spiderweb fractures
        edges_fine = cv2.Canny(filtered_gray, 30, 100)
        edges_coarse = cv2.Canny(gray, 60, 180)
        combined_edges = cv2.bitwise_or(edges_fine, edges_coarse)

        # Suppress edges that fall inside smooth glare saturated zones without structure
        # (keeps genuine crack edges traversing across glare)
        glare_eroded = cv2.erode(glare_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        valid_edges = cv2.bitwise_and(combined_edges, cv2.bitwise_not(glare_eroded))

        # Compute edge orientation entropy
        entropy = self.compute_gradient_entropy(gray, valid_edges)

        # Micro-fracture edge density in effective analyzed area
        non_glare_pixels = max(1, (h * w) - np.count_nonzero(glare_eroded))
        fracture_density = float(np.count_nonzero(valid_edges) / non_glare_pixels)

        # Count connected crack branches / fragment components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid_edges, connectivity=8)
        # Exclude background label 0 and single-pixel noise
        valid_components = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 4]
        crack_branch_count = len(valid_components)

        return {
            "fracture_density": fracture_density,
            "gradient_entropy": entropy,
            "laplacian_variance": laplacian_var,
            "glare_ratio": glare_ratio,
            "fracture_mask": valid_edges,
            "crack_branch_count": crack_branch_count
        }

    def generate_fracture_heatmap(
        self,
        bgr_crop: np.ndarray,
        fracture_mask: np.ndarray
    ) -> np.ndarray:
        """
        Creates an illuminated fracture intensity heatmap overlay on the BGR crop.
        Returns RGB image ready for display.
        """
        h, w = bgr_crop.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((10, 10, 3), dtype=np.uint8)

        # Distance transform / Gaussian blur on edge mask to create heat diffusion
        dist = cv2.distanceTransform((fracture_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        # Invert distance so crack centers are hottest
        heat = np.exp(-dist / 3.5)
        heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)

        # Apply vibrant JET / INFERNO colormap
        heatmap_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

        # Blend with original image
        alpha = 0.55
        blended_bgr = cv2.addWeighted(bgr_crop, 1.0 - alpha, heatmap_bgr, alpha, 0)

        # Highlight sharpest fracture lines in bright cyan / yellow
        blended_bgr[fracture_mask > 0] = (255, 230, 0)

        return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    def analyze_window_crop(
        self,
        bgr_crop: np.ndarray,
        detector_conf: float = 0.50,
        sensitivity: float = 0.50
    ) -> GlassDiagnosticResult:
        """
        Evaluates a window crop to verify whether the glass is Shattered, Cracked, or Intact.
        - sensitivity: [0.0, 1.0] (higher = more sensitive to hairline fractures, lower = stricter false-positive rejection)
        """
        feats = self.compute_fracture_features(bgr_crop)
        f_dens = feats["fracture_density"]
        entropy = feats["gradient_entropy"]
        lap_var = feats["laplacian_variance"]
        glare_r = feats["glare_ratio"]
        f_mask = feats["fracture_mask"]
        branches = feats["crack_branch_count"]

        # Sensitivity adjustments
        # Adjust density and entropy thresholds based on sensitivity
        adj_dens_thresh = self.fracture_density_thresh * (1.5 - sensitivity)
        adj_entropy_thresh = self.entropy_thresh * (1.3 - 0.6 * sensitivity)

        # Heuristic scoring components
        # 1. Density score [0 - 1]
        score_density = np.clip(f_dens / max(1e-4, adj_dens_thresh * 2.0), 0.0, 1.0)

        # 2. Entropy score (multi-directional fracture network) [0 - 1]
        score_entropy = np.clip((entropy - 0.20) / 0.60, 0.0, 1.0)

        # 3. Laplacian texture sharpness score [0 - 1]
        score_laplacian = np.clip(lap_var / (self.laplacian_thresh * 2.5), 0.0, 1.0)

        # 4. Glare penalty: if large glare blob without high entropy, it's a reflection
        glare_penalty = 0.0
        if glare_r > 0.25 and entropy < 0.65:
            glare_penalty = 0.35 * (glare_r / 0.5)

        # Weighted composite shatter confidence score
        raw_shatter_score = (
            0.40 * score_density +
            0.35 * score_entropy +
            0.15 * score_laplacian +
            0.10 * np.clip(branches / 30.0, 0.0, 1.0) -
            glare_penalty
        )
        shatter_conf = float(np.clip(raw_shatter_score * 0.70 + detector_conf * 0.30, 0.0, 1.0))

        # Classification Logic
        is_shattered = False
        status = "INTACT"
        reason = ""

        # Condition 1: Dense spiderweb shatter (dicing or heavy fracture)
        if f_dens >= adj_dens_thresh and entropy >= adj_entropy_thresh:
            is_shattered = True
            status = "SHATTERED"
            shatter_conf = max(shatter_conf, 0.75)
            reason = f"High-density omnidirectional fracture network detected (density: {f_dens*100:.1f}%, entropy: {entropy:.2f}, branches: {branches})."

        # Condition 2: Linear / radiating crack or localized impact puncture
        elif f_dens >= (adj_dens_thresh * 0.65) and (lap_var > self.laplacian_thresh or branches >= 12):
            is_shattered = True
            status = "CRACKED"
            shatter_conf = max(shatter_conf, 0.60)
            reason = f"Structural crack / fracture line verified across glass surface (density: {f_dens*100:.1f}%, branches: {branches})."

        # Condition 3: Sun glare / reflection false positive
        elif glare_r > 0.20 and entropy < 0.50:
            is_shattered = False
            status = "INTACT"
            shatter_conf = min(shatter_conf, 0.25)
            reason = f"Intact window profile: Specular sun glare / horizon reflection without omnidirectional fracture network (glare: {glare_r*100:.0f}%, entropy: {entropy:.2f})."

        # Condition 4: Clean intact window / tinted glass
        else:
            is_shattered = False
            status = "INTACT"
            shatter_conf = min(shatter_conf, 0.30)
            reason = f"Intact window surface: Smooth gradient profile with low fracture texture density ({f_dens*100:.2f}%)."

        # Generate visual heatmap overlay
        heatmap_rgb = self.generate_fracture_heatmap(bgr_crop, f_mask)

        return GlassDiagnosticResult(
            status=status,
            is_shattered_or_cracked=is_shattered,
            shatter_confidence=round(shatter_conf, 3),
            fracture_density=round(f_dens, 4),
            gradient_entropy=round(entropy, 3),
            laplacian_variance=round(lap_var, 2),
            glare_ratio=round(glare_r, 3),
            reason=reason,
            fracture_heatmap_rgb=heatmap_rgb
        )
