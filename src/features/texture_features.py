"""
Texture Analysis Feature Extraction Module for Vehicle Damage Assessment.

Educational Note — Why Texture Analysis Detects Vehicle Damage:
--------------------------------------------------------------
Vehicle sheet metal, clear-coats, and intact glass have smooth, homogeneous, low-variance textures.
When physical damage occurs, texture changes fundamentally:
1. Gray-Level Co-occurrence Matrix (GLCM):
   - GLCM computes how frequently a pixel with grayscale value i occurs adjacent to a pixel with value j
     at specified distances (d) and angles (0°, 45°, 90°, 135°).
   - *Contrast* & *Dissimilarity*: High when adjacent pixels have sharply different values (scratches, cracks).
   - *Homogeneity*: Measures smoothness. Pristine factory panels have near-1.0 homogeneity; crumpled metal has low values.
   - *Energy (Uniformity)*: Measures orderliness in pixel transitions.
   - *Correlation*: Measures linear dependency between neighbor pixels.
2. Local Binary Patterns (LBP):
   - For each pixel, LBP compares it against its P circular neighbors at radius R.
   - If neighbor >= center, bit is 1; else 0. This forms a binary number (e.g. 11001001).
   - 'Uniform' LBPs have at most 2 bitwise 0->1 or 1->0 transitions (representing fundamental micro-edges, spots, corners).
   - Shattered glass and torn bumpers produce rich high-entropy LBP distributions, while clean panels concentrate in uniform flat bins.
3. Gabor Filter Banks:
   - 2D Gaussian modulated by a sinusoidal plane wave at specific spatial frequencies (wavelength lambda)
     and orientations (theta).
   - Directly detects directional fracture networks (e.g. diagonal bumper scrapes vs horizontal panel creases).
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from typing import Dict, List, Tuple


def get_texture_feature_names() -> Dict[str, str]:
    """
    Returns a dictionary mapping every texture feature column name
    to its human-readable one-line description (48 features total).
    """
    descriptions = {}

    # 1. GLCM features (6 properties * 2 summary stats = 12 features)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        descriptions[f"texture_glcm_{prop}_mean"] = f"GLCM {prop.capitalize()} mean across distances (1, 3px) and 4 angles."
        descriptions[f"texture_glcm_{prop}_std"] = f"GLCM {prop.capitalize()} standard deviation across angles."

    # 2. Local Binary Patterns (P=8, R=1: 10 uniform bins + mean + entropy = 12 features)
    for b in range(10):
        descriptions[f"texture_lbp_bin{b}"] = f"Uniform LBP (radius=1, neighbors=8) micro-texture histogram bin #{b}."
    descriptions["texture_lbp_mean"] = "Mean intensity of LBP texture pattern map."
    descriptions["texture_lbp_entropy"] = "Shannon entropy of LBP micro-texture distribution."

    # 3. Gabor Filter Bank (4 angles x 3 wavelengths = 12 filters x 2 stats = 24 features)
    angles = [0, 45, 90, 135]
    wavelengths = [4, 8, 16]
    for theta in angles:
        for lam in wavelengths:
            descriptions[f"texture_gabor_th{theta}_lam{lam}_mean"] = f"Gabor response energy (orientation={theta}°, wavelength={lam}px)."
            descriptions[f"texture_gabor_th{theta}_lam{lam}_std"] = f"Gabor response variance (orientation={theta}°, wavelength={lam}px)."

    return descriptions


def extract_texture_features(rgb_img: np.ndarray) -> Dict[str, float]:
    """
    Extracts a fixed 48-dimensional texture feature vector from an image.

    Parameters:
    -----------
    rgb_img : NumPy RGB image array [H, W, 3]

    Returns:
    --------
    Dict[str, float] of exactly 48 documented numeric texture features.
    """
    feats: Dict[str, float] = {}
    names = get_texture_feature_names()

    if rgb_img is None or rgb_img.size == 0:
        for k in names.keys():
            feats[k] = 0.0
        return feats

    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    gray_std = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------
    # 1. Gray-Level Co-occurrence Matrix (GLCM) (12 features)
    # -------------------------------------------------------------
    gray_32 = (gray_std // 8).astype(np.uint8)
    distances = [1, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    try:
        glcm = graycomatrix(
            gray_32,
            distances=distances,
            angles=angles,
            levels=32,
            symmetric=True,
            normed=True
        )

        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
            vals = graycoprops(glcm, prop)
            vals_clean = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=-1.0)
            feats[f"texture_glcm_{prop}_mean"] = float(np.mean(vals_clean))
            feats[f"texture_glcm_{prop}_std"] = float(np.std(vals_clean))
    except Exception:
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
            feats[f"texture_glcm_{prop}_mean"] = 0.0
            feats[f"texture_glcm_{prop}_std"] = 0.0

    # -------------------------------------------------------------
    # 2. Local Binary Patterns (LBP) (12 features)
    # -------------------------------------------------------------
    try:
        lbp_r1 = local_binary_pattern(gray_std, P=8, R=1, method="uniform")
        n_bins = 10
        hist, _ = np.histogram(lbp_r1.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        for b in range(10):
            feats[f"texture_lbp_bin{b}"] = float(hist[b])

        feats["texture_lbp_mean"] = float(np.mean(lbp_r1)) / float(n_bins)

        p = hist[hist > 0]
        entropy = -np.sum(p * np.log2(p)) / np.log2(n_bins)
        feats["texture_lbp_entropy"] = float(np.clip(entropy, 0.0, 1.0))
    except Exception:
        for b in range(10):
            feats[f"texture_lbp_bin{b}"] = 0.0
        feats["texture_lbp_mean"] = 0.0
        feats["texture_lbp_entropy"] = 0.0

    # -------------------------------------------------------------
    # 3. Gabor Filter Bank Responses (24 features)
    # -------------------------------------------------------------
    angles_deg = [0, 45, 90, 135]
    wavelengths = [4, 8, 16]
    gray_float = gray_std.astype(np.float32) / 255.0

    for theta_deg in angles_deg:
        theta_rad = np.deg2rad(theta_deg)
        for lam in wavelengths:
            kernel = cv2.getGaborKernel(
                ksize=(15, 15),
                sigma=2.5,
                theta=theta_rad,
                lambd=float(lam),
                gamma=0.5,
                psi=0.0,
                ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray_float, cv2.CV_32F, kernel)
            energy = np.abs(filtered)

            feats[f"texture_gabor_th{theta_deg}_lam{lam}_mean"] = float(np.mean(energy))
            feats[f"texture_gabor_th{theta_deg}_lam{lam}_std"] = float(np.std(energy))

    return feats
