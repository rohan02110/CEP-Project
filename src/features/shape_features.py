"""
Shape, Edge, Contour, Frequency & Reflectance Feature Extraction Module.

Educational Note — Physical Interpretation of Shape & Frequency Features:
--------------------------------------------------------------------------
1. Canny Edge Density & Orientation Entropy:
   - Clean car panels produce few, linear structural edges (door seams, rooflines).
   - Crumpled collision damage and shattered glass produce dense, omnidirectional spiderweb edges.
   - Gradient Orientation Entropy measures edge direction randomness: high for shattered glass, low for horizontal body creases.
2. Contour Circularity (4 * pi * Area / Perimeter^2):
   - Perfect circles have a circularity of 1.0; complex jagged shapes approach 0.0.
   - Tire Deflation Principle: An inflated wheel is round and circular (circularity ~0.65 - 0.95).
     A deflated/flat tire collapses under vehicle weight, flattening against the tarmac (circularity < 0.40).
3. Solidity & Convexity Defects:
   - Solidity = Contour Area / Convex Hull Area.
   - Intact panels are smooth and convex (solidity ~0.90 - 1.0).
   - Deep structural dents and smashed bumpers create large inward folds, increasing convexity defects and dropping solidity.
4. 2D Fast Fourier Transform (FFT) Frequency Analysis:
   - 2D FFT shifts the image from spatial coordinates (x, y) into 2D spatial frequencies (u, v).
   - Low spatial frequencies represent broad, gradual illumination changes (intact metal panels, wide shallow dents).
   - High spatial frequencies represent sharp, rapid pixel transitions (hairline scratches, shattered tempered glass).
   - The High-to-Low Frequency Energy Ratio serves as a physical separator between scratches and dents.
5. Specular Glare & Reflectance Masking:
   - Car clear-coats and windshields create bright white sun glares (Value > 215, Saturation < 65).
   - Measuring the Specular Highlight Ratio allows the model to distinguish harmless sun reflections from actual cracks.
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple


def get_shape_feature_names() -> Dict[str, str]:
    """
    Returns a dictionary mapping every shape/frequency/reflectance feature column name
    to its human-readable one-line description.
    """
    return {
        # 1. Edge & Gradient features (5 features)
        "shape_canny_density_fine": "Fine Canny edge pixel density (thresholds: 30, 90).",
        "shape_canny_density_coarse": "Coarse Canny edge pixel density (thresholds: 70, 180).",
        "shape_sobel_gradient_mean": "Mean Sobel gradient magnitude across the entire frame.",
        "shape_sobel_gradient_std": "Standard deviation of Sobel gradient magnitude.",
        "shape_edge_orientation_entropy": "Shannon entropy of edge gradient directions (0=unidirectional, 1=isotropic shatter).",

        # 2. Contour & Geometric Shape features (8 features)
        "shape_contour_count": "Number of distinct external contours detected above noise threshold.",
        "shape_contour_mean_area_ratio": "Mean contour area normalized by total image area.",
        "shape_contour_max_area_ratio": "Largest individual contour area normalized by total image area.",
        "shape_contour_area_to_perimeter": "Ratio of total contour area to total perimeter length.",
        "shape_circularity_primary": "Isoperimetric circularity quotient (4*pi*A/P^2) of primary contour (1.0=circle, <0.4=deflated/crumpled).",
        "shape_solidity_primary": "Solidity ratio (Contour Area / Convex Hull Area) of primary contour.",
        "shape_aspect_ratio_primary": "Bounding box aspect ratio (Width / Height) of primary contour.",
        "shape_convexity_defects_count": "Number of significant inward structural convexity defect cavities.",

        # 3. Frequency Domain Features via 2D FFT (5 features)
        "shape_fft_total_spectral_energy": "Total logarithmic power spectrum energy from 2D Fast Fourier Transform.",
        "shape_fft_low_freq_ratio": "Fraction of spectral energy concentrated in low spatial frequencies (<20% Nyquist).",
        "shape_fft_high_freq_ratio": "Fraction of spectral energy concentrated in high spatial frequencies (>50% Nyquist).",
        "shape_fft_high_to_low_ratio": "Ratio of high-frequency to low-frequency energy (separates scratches from broad dents).",
        "shape_fft_spectral_centroid": "Radial spectral centroid (average spatial frequency wavelength).",

        # 4. Reflectance & Glare Features (5 features)
        "shape_specular_glare_ratio": "Fraction of image pixels occupied by specular sun glare (Value>215, Saturation<65).",
        "shape_specular_mean_val": "Mean brightness value inside detected specular highlight zones.",
        "shape_specular_mean_sat": "Mean saturation purity inside detected specular highlight zones.",
        "shape_glare_suppressed_edge_density": "Edge density strictly outside specular glare reflections.",
        "shape_damage_blob_count": "Number of localized high-gradient candidate damage blob clusters."
    }


def extract_shape_features(rgb_img: np.ndarray) -> Dict[str, float]:
    """
    Extracts a fixed 23-dimensional shape, contour, frequency, and reflectance feature vector.

    Parameters:
    -----------
    rgb_img : NumPy RGB image array [H, W, 3]

    Returns:
    --------
    Dict[str, float] of documented numeric shape & frequency features.
    """
    feats: Dict[str, float] = {}
    names = get_shape_feature_names()

    if rgb_img is None or rgb_img.size == 0:
        for k in names.keys():
            feats[k] = 0.0
        return feats

    h, w = rgb_img.shape[:2]
    img_area = float(max(1, h * w))
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # -------------------------------------------------------------
    # 1. Edge & Gradient Metrics
    # -------------------------------------------------------------
    # Fine & coarse multi-scale Canny edge maps
    edges_fine = cv2.Canny(gray, 30, 90)
    edges_coarse = cv2.Canny(gray, 70, 180)

    feats["shape_canny_density_fine"] = float(np.count_nonzero(edges_fine) / img_area)
    feats["shape_canny_density_coarse"] = float(np.count_nonzero(edges_coarse) / img_area)

    # Sobel gradient magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)

    feats["shape_sobel_gradient_mean"] = float(np.mean(sobel_mag) / 255.0)
    feats["shape_sobel_gradient_std"] = float(np.std(sobel_mag) / 255.0)

    # Edge Orientation Entropy
    edge_mask = edges_fine > 0
    if np.count_nonzero(edge_mask) > 20:
        angles = np.arctan2(sobel_y, sobel_x) * (180.0 / np.pi)
        angles = np.mod(angles, 180.0)
        edge_angles = angles[edge_mask]
        hist, _ = np.histogram(edge_angles, bins=18, range=(0, 180), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        norm_entropy = float(np.clip(entropy / np.log2(18.0), 0.0, 1.0))
    else:
        norm_entropy = 0.0
    feats["shape_edge_orientation_entropy"] = norm_entropy

    # -------------------------------------------------------------
    # 2. Contour Topology & Circularity
    # -------------------------------------------------------------
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [c for c in contours if cv2.contourArea(c) > 25.0]
    feats["shape_contour_count"] = float(len(valid_contours))

    if len(valid_contours) > 0:
        areas = [cv2.contourArea(c) for c in valid_contours]
        perims = [cv2.arcLength(c, True) for c in valid_contours]

        feats["shape_contour_mean_area_ratio"] = float(np.mean(areas) / img_area)
        feats["shape_contour_max_area_ratio"] = float(np.max(areas) / img_area)

        tot_area = sum(areas)
        tot_perim = sum(perims)
        feats["shape_contour_area_to_perimeter"] = float(tot_area / max(1.0, tot_perim))

        # Primary (largest) contour circularity & solidity
        primary_c = max(valid_contours, key=cv2.contourArea)
        p_area = cv2.contourArea(primary_c)
        p_perim = cv2.arcLength(primary_c, True)

        if p_perim > 0:
            circularity = 4.0 * np.pi * (p_area / (p_perim * p_perim))
            feats["shape_circularity_primary"] = float(np.clip(circularity, 0.0, 1.0))
        else:
            feats["shape_circularity_primary"] = 0.0

        # Convex Hull & Solidity
        hull = cv2.convexHull(primary_c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            feats["shape_solidity_primary"] = float(np.clip(p_area / hull_area, 0.0, 1.0))
        else:
            feats["shape_solidity_primary"] = 1.0

        # Bounding box aspect ratio
        _, _, bw, bh = cv2.boundingRect(primary_c)
        feats["shape_aspect_ratio_primary"] = float(bw / max(1.0, bh))

        # Convexity defects (structural indentations & crumpled folds)
        if len(primary_c) >= 5:
            try:
                hull_indices = cv2.convexHull(primary_c, returnPoints=False)
                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(primary_c, hull_indices)
                    if defects is not None:
                        # Count defects with depth > 5 pixels (scaled by 256 in OpenCV)
                        sig_defects = [d for d in defects if d[0][3] > (5 * 256)]
                        feats["shape_convexity_defects_count"] = float(len(sig_defects))
                    else:
                        feats["shape_convexity_defects_count"] = 0.0
                else:
                    feats["shape_convexity_defects_count"] = 0.0
            except Exception:
                feats["shape_convexity_defects_count"] = 0.0
        else:
            feats["shape_convexity_defects_count"] = 0.0
    else:
        feats["shape_contour_mean_area_ratio"] = 0.0
        feats["shape_contour_max_area_ratio"] = 0.0
        feats["shape_contour_area_to_perimeter"] = 0.0
        feats["shape_circularity_primary"] = 0.0
        feats["shape_solidity_primary"] = 0.0
        feats["shape_aspect_ratio_primary"] = 1.0
        feats["shape_convexity_defects_count"] = 0.0

    # -------------------------------------------------------------
    # 3. 2D Fast Fourier Transform (FFT) Frequency Domain
    # -------------------------------------------------------------
    # Standardize resolution for frequency spectrum
    gray_fft = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    
    # Compute 2D FFT & shift zero frequency to center
    fft_complex = np.fft.fft2(gray_fft)
    fft_shifted = np.fft.fftshift(fft_complex)
    magnitude_spectrum = np.abs(fft_shifted)

    total_spectral_energy = np.sum(magnitude_spectrum)
    feats["shape_fft_total_spectral_energy"] = float(np.log1p(total_spectral_energy))

    # Compute radial distance matrix from center (radius r in [0, 64])
    cy, cx = 64, 64
    y_coords, x_coords = np.ogrid[:128, :128]
    r_matrix = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

    # Low frequencies: r < 12 (approx <20% of Nyquist)
    low_freq_mask = r_matrix < 12.0
    low_energy = np.sum(magnitude_spectrum[low_freq_mask])
    
    # High frequencies: r > 32 (approx >50% of Nyquist)
    high_freq_mask = r_matrix > 32.0
    high_energy = np.sum(magnitude_spectrum[high_freq_mask])

    if total_spectral_energy > 0:
        feats["shape_fft_low_freq_ratio"] = float(low_energy / total_spectral_energy)
        feats["shape_fft_high_freq_ratio"] = float(high_energy / total_spectral_energy)
    else:
        feats["shape_fft_low_freq_ratio"] = 0.0
        feats["shape_fft_high_freq_ratio"] = 0.0

    # High-to-Low Ratio (Scratch vs Dent separator)
    feats["shape_fft_high_to_low_ratio"] = float(high_energy / max(1.0, low_energy))

    # Spectral Centroid (mean radial frequency)
    if total_spectral_energy > 0:
        spectral_centroid = np.sum(r_matrix * magnitude_spectrum) / (total_spectral_energy * 64.0)
        feats["shape_fft_spectral_centroid"] = float(np.clip(spectral_centroid, 0.0, 1.0))
    else:
        feats["shape_fft_spectral_centroid"] = 0.0

    # -------------------------------------------------------------
    # 4. Reflectance & Specular Glare Analysis
    # -------------------------------------------------------------
    v_chan = hsv[:, :, 2]
    s_chan = hsv[:, :, 1]

    # Specular highlights: high brightness (V > 215) and low saturation (S < 65)
    glare_mask = (v_chan > 215) & (s_chan < 65)
    glare_pixels = np.count_nonzero(glare_mask)
    feats["shape_specular_glare_ratio"] = float(glare_pixels / img_area)

    if glare_pixels > 0:
        feats["shape_specular_mean_val"] = float(np.mean(v_chan[glare_mask]) / 255.0)
        feats["shape_specular_mean_sat"] = float(np.mean(s_chan[glare_mask]) / 255.0)
    else:
        feats["shape_specular_mean_val"] = 0.0
        feats["shape_specular_mean_sat"] = 0.0

    # Edge density outside glare
    non_glare_mask = ~glare_mask
    non_glare_pixels = np.count_nonzero(non_glare_mask)
    if non_glare_pixels > 0:
        edges_non_glare = np.count_nonzero(edges_fine & non_glare_mask)
        feats["shape_glare_suppressed_edge_density"] = float(edges_non_glare / float(non_glare_pixels))
    else:
        feats["shape_glare_suppressed_edge_density"] = 0.0

    # Localized high-gradient damage blob count
    _, high_grad = cv2.threshold(sobel_mag.astype(np.uint8), 60, 255, cv2.THRESH_BINARY)
    num_blobs, _, stats, _ = cv2.connectedComponentsWithStats(high_grad, connectivity=8)
    # Exclude background blob 0 and tiny noise (<10 pixels)
    valid_blobs = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 10]
    feats["shape_damage_blob_count"] = float(len(valid_blobs))

    return feats
