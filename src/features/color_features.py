"""
Color Space Feature Extraction Module for Vehicle Damage Assessment.

Educational Note — Why Color Features Matter for Damage Diagnosis:
-------------------------------------------------------------------
1. RGB vs. HSV Color Spaces:
   - RGB (Red, Green, Blue) coordinates are highly coupled with illumination intensity.
     A shadow on a red car shifts all three channels simultaneously.
   - HSV (Hue, Saturation, Value) decouples chromatic color (Hue) from lighting brightness (Value)
     and color purity (Saturation). When paint scratches expose grey metallic primer, or when glass
     shatters into white micro-reflections, the Hue and Saturation shift dramatically while Value
     spikes, creating distinct signatures.
2. Statistical Color Moments (Mean, Std):
   - Mean indicates overall tonal lightness in each color channel.
   - Standard Deviation indicates color contrast (scratches and cracked lamps create local color variance).
3. Dominant Color K-Means Clustering:
   - Clusters image pixels into K representative color centroids.
   - For an intact car, one dominant centroid occupies ~80%+ of pixels. When severe structural damage
     occurs (exposed metal, black tires, cracked lenses), secondary and tertiary cluster proportions increase.
"""

import cv2
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple


def get_color_feature_names() -> Dict[str, str]:
    """
    Returns a dictionary mapping every color feature column name
    to its human-readable one-line description (48 features total).
    """
    descriptions = {}
    
    # 1. RGB Histograms (4 bins per channel = 12 features)
    for c in ["r", "g", "b"]:
        for b in range(4):
            descriptions[f"color_hist_{c}_bin{b}"] = f"Normalized pixel frequency in {c.upper()} channel intensity bin {b}/4."

    # 2. HSV Histograms (4 bins per channel = 12 features)
    for c in ["h", "s", "v"]:
        for b in range(4):
            descriptions[f"color_hist_{c}_bin{b}"] = f"Normalized pixel frequency in {c.upper()} channel intensity bin {b}/4."

    # 3. Per-channel statistical moments (Mean & Std across 6 channels = 12 features)
    for ch in ["r", "g", "b", "h", "s", "v"]:
        descriptions[f"color_{ch}_mean"] = f"Mean pixel intensity in {ch.upper()} channel."
        descriptions[f"color_{ch}_std"] = f"Standard deviation of pixel intensity in {ch.upper()} channel."

    # 4. K-Means Dominant Colors (3 centroids * 3 channels + 3 proportions = 12 features)
    for k in range(3):
        descriptions[f"color_dominant_c{k}_r"] = f"Red intensity of dominant color centroid #{k+1}."
        descriptions[f"color_dominant_c{k}_g"] = f"Green intensity of dominant color centroid #{k+1}."
        descriptions[f"color_dominant_c{k}_b"] = f"Blue intensity of dominant color centroid #{k+1}."
        descriptions[f"color_dominant_c{k}_prop"] = f"Pixel area fraction belonging to dominant color cluster #{k+1}."

    return descriptions


def extract_color_features(rgb_img: np.ndarray) -> Dict[str, float]:
    """
    Extracts a fixed 48-dimensional color feature vector from an RGB image.

    Parameters:
    -----------
    rgb_img : NumPy RGB image array [H, W, 3]

    Returns:
    --------
    Dict[str, float] of exactly 48 documented numeric color features.
    """
    feats: Dict[str, float] = {}
    names = get_color_feature_names()

    if rgb_img is None or rgb_img.size == 0:
        for name in names.keys():
            feats[name] = 0.0
        return feats

    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    total_pixels = float(rgb_img.shape[0] * rgb_img.shape[1])

    # -------------------------------------------------------------
    # 1. RGB Color Histograms (4 bins per channel, normalized)
    # -------------------------------------------------------------
    for c_idx, c_name in enumerate(["r", "g", "b"]):
        hist = cv2.calcHist([rgb_img], [c_idx], None, [4], [0, 256]).flatten()
        hist_norm = hist / max(1.0, total_pixels)
        for b in range(4):
            feats[f"color_hist_{c_name}_bin{b}"] = float(hist_norm[b])

    # -------------------------------------------------------------
    # 2. HSV Color Histograms (4 bins per channel, normalized)
    # -------------------------------------------------------------
    hist_h = cv2.calcHist([hsv_img], [0], None, [4], [0, 180]).flatten() / max(1.0, total_pixels)
    for b in range(4):
        feats[f"color_hist_h_bin{b}"] = float(hist_h[b])

    hist_s = cv2.calcHist([hsv_img], [1], None, [4], [0, 256]).flatten() / max(1.0, total_pixels)
    for b in range(4):
        feats[f"color_hist_s_bin{b}"] = float(hist_s[b])

    hist_v = cv2.calcHist([hsv_img], [2], None, [4], [0, 256]).flatten() / max(1.0, total_pixels)
    for b in range(4):
        feats[f"color_hist_v_bin{b}"] = float(hist_v[b])

    # -------------------------------------------------------------
    # 3. Statistical Moments (Mean & Std) per channel
    # -------------------------------------------------------------
    channels = {
        "r": rgb_img[:, :, 0].astype(np.float32) / 255.0,
        "g": rgb_img[:, :, 1].astype(np.float32) / 255.0,
        "b": rgb_img[:, :, 2].astype(np.float32) / 255.0,
        "h": hsv_img[:, :, 0].astype(np.float32) / 180.0,
        "s": hsv_img[:, :, 1].astype(np.float32) / 255.0,
        "v": hsv_img[:, :, 2].astype(np.float32) / 255.0,
    }

    for ch_name, ch_data in channels.items():
        flat = ch_data.flatten()
        feats[f"color_{ch_name}_mean"] = float(np.mean(flat))
        feats[f"color_{ch_name}_std"] = float(np.std(flat))

    # -------------------------------------------------------------
    # 4. K-Means Dominant Color Clustering (k=3)
    # -------------------------------------------------------------
    small_rgb = cv2.resize(rgb_img, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = small_rgb.reshape(-1, 3).astype(np.float32) / 255.0

    # If the image is flat/monochrome (very low variance across pixels), avoid KMeans warning
    if float(np.std(pixels)) < 1e-4:
        mean_c = np.mean(pixels, axis=0)
        for k_idx in range(3):
            feats[f"color_dominant_c{k_idx}_r"] = float(mean_c[0])
            feats[f"color_dominant_c{k_idx}_g"] = float(mean_c[1])
            feats[f"color_dominant_c{k_idx}_b"] = float(mean_c[2])
            feats[f"color_dominant_c{k_idx}_prop"] = 1.0 if k_idx == 0 else 0.0
    else:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=3, max_iter=100)
                labels = kmeans.fit_predict(pixels)
                centers = kmeans.cluster_centers_

            counts = np.bincount(labels, minlength=3)
            props = counts / float(len(labels))
            sort_order = np.argsort(-props)

            for k_idx, cluster_i in enumerate(sort_order):
                c_r, c_g, c_b = centers[cluster_i]
                p_val = props[cluster_i]
                feats[f"color_dominant_c{k_idx}_r"] = float(c_r)
                feats[f"color_dominant_c{k_idx}_g"] = float(c_g)
                feats[f"color_dominant_c{k_idx}_b"] = float(c_b)
                feats[f"color_dominant_c{k_idx}_prop"] = float(p_val)
        except Exception:
            for k_idx in range(3):
                feats[f"color_dominant_c{k_idx}_r"] = 0.0
                feats[f"color_dominant_c{k_idx}_g"] = 0.0
                feats[f"color_dominant_c{k_idx}_b"] = 0.0
                feats[f"color_dominant_c{k_idx}_prop"] = 0.333

    return feats
