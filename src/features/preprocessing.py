"""
Classical Computer Vision Preprocessing & Region Proposal Module.

Educational Note — Why Preprocessing & Classical Region Proposals Matter:
--------------------------------------------------------------------------
In classical computer vision, we do not have deep convolutional layers that learn
invariance automatically. Instead, we explicitly condition the image:
1. Bilateral Filtering: Unlike standard Gaussian smoothing which blurs all pixels equally,
   a bilateral filter averages pixels based on both spatial closeness and color similarity.
   This removes high-frequency sensor noise while keeping structural edges (like scratch lines
   and panel seams) crisp and sharp.
2. Morphological Operations: Opening (erosion followed by dilation) eliminates small background
   speckles; Closing (dilation followed by erosion) bridges small gaps in vehicle contours.
3. Otsu & Adaptive Thresholding: Automatically finds the optimal luminance boundary between the
   vehicle foreground and the background without relying on fixed arbitrary thresholds.
4. Classical Region Proposals: By identifying clusters of high gradient magnitudes (Canny/Sobel)
   and localized contour anomalies, we propose candidate damage zones without needing a deep
   object detection network like YOLO.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path


def load_image_rgb(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Loads and standardizes an image to an 8-bit RGB NumPy array [H, W, 3].
    Handles filepaths, PIL Images, and BGR/RGB NumPy arrays safely.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
        return np.array(img, dtype=np.uint8)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:  # Grayscale
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 4:  # RGBA
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.shape[2] == 3:
                return image_input.copy()
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def preprocess_image(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (512, 512),
    denoise_method: str = "bilateral"
) -> np.ndarray:
    """
    Standardizes image dimensions, enhances contrast, and applies edge-preserving denoising.

    Parameters:
    -----------
    image_input : file path, PIL image, or numpy array
    target_size : (width, height) to resize standardizing scale across varied resolutions
    denoise_method : 'bilateral' for edge-preserving smoothing, 'gaussian' for standard blur, or None

    Returns:
    --------
    np.ndarray : Cleaned RGB image of shape (target_size[1], target_size[0], 3)
    """
    rgb = load_image_rgb(image_input)
    
    # Resize with high-quality area/cubic interpolation
    if target_size is not None:
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA if (rgb.shape[1] > target_size[0]) else cv2.INTER_CUBIC)

    # Apply edge-preserving bilateral filtering
    if denoise_method == "bilateral":
        # d=5: neighborhood diameter; sigmaColor=50: color mix distance; sigmaSpace=50: spatial distance
        rgb = cv2.bilateralFilter(rgb, d=5, sigmaColor=50, sigmaSpace=50)
    elif denoise_method == "gaussian":
        rgb = cv2.GaussianBlur(rgb, (5, 5), 0)

    return rgb


def isolate_vehicle_roi(
    rgb_img: np.ndarray,
    min_area_ratio: float = 0.15
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Classical Foreground Vehicle Isolation using Adaptive Thresholding & Convex Contours.
    Eliminates irrelevant workshop floors, sky, and distant peripheral background clutter.

    Parameters:
    -----------
    rgb_img : RGB image array [H, W, 3]
    min_area_ratio : minimum fraction of image area required to consider a bounding box valid

    Returns:
    --------
    cropped_roi : The extracted foreground vehicle RGB crop
    bbox : (x1, y1, x2, y2) pixel bounding coordinates in the original image
    """
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # 1. Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Otsu thresholding + Canny edge boundary detection
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(enhanced, 40, 120)
    combined = cv2.bitwise_or(otsu, edges)

    # 3. Morphological closing to seal car silhouette contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # 4. Find external contours and select the largest central foreground contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = float(h * w)
    best_bbox = (0, 0, w, h)
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if area / img_area < min_area_ratio:
            continue
        
        x, y, bw, bh = cv2.boundingRect(c)
        cx = x + (bw / 2.0)
        cy = y + (bh / 2.0)
        
        # Prefer large central objects (subject vehicle)
        dist_from_center = np.sqrt(((cx - w/2.0) / (w/2.0))**2 + ((cy - h/2.0) / (h/2.0))**2)
        score = (area / img_area) * (1.0 - 0.4 * min(1.0, dist_from_center))

        if score > best_score:
            best_score = score
            # Add 3% margin to ensure edge damage is not cut off
            pad_x = int(bw * 0.03)
            pad_y = int(bh * 0.03)
            best_bbox = (
                max(0, x - pad_x),
                max(0, y - pad_y),
                min(w, x + bw + pad_x),
                min(h, y + bh + pad_y)
            )

    x1, y1, x2, y2 = best_bbox
    cropped_roi = rgb_img[y1:y2, x1:x2]
    return cropped_roi, best_bbox


def propose_damage_regions(
    rgb_img: np.ndarray,
    max_regions: int = 5
) -> List[Dict[str, Union[np.ndarray, Tuple[int, int, int, int], float]]]:
    """
    Classical CV Damage Candidate Region Proposal.
    Finds localized high-gradient disturbance regions (scratches, dents, cracks, punctures)
    using Sobel gradient concentration, adaptive thresholding, and morphological clustering.
    This replaces YOLO's localization role with a purely inspectable, classical CV method.

    Returns:
    --------
    List of proposed region dictionaries containing:
        - 'bbox': (x1, y1, x2, y2)
        - 'crop': RGB crop of the candidate region
        - 'area_ratio': region area relative to image
        - 'gradient_energy': mean edge gradient energy inside region
    """
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    img_area = float(h * w)

    # 1. Compute Sobel gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Threshold high-gradient disturbances (scratches, cracks, jagged edges)
    _, high_grad_mask = cv2.threshold(grad_norm, 60, 255, cv2.THRESH_BINARY)

    # 3. Morphological dilation to connect nearby fracture branches & dent crease lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.morphologyEx(high_grad_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Find bounding boxes for damage candidates
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        area_ratio = area / img_area
        # Filter tiny speckle noise (<0.1%) and massive background covers (>60%)
        if area_ratio < 0.001 or area_ratio > 0.60:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        # Add 5% padding around candidate damage zone
        px = int(bw * 0.05)
        py = int(bh * 0.05)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(w, x + bw + px), min(h, y + bh + py)

        crop = rgb_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_grad = grad_norm[y1:y2, x1:x2]
        mean_grad = float(np.mean(crop_grad))

        candidates.append({
            "bbox": (x1, y1, x2, y2),
            "crop": crop,
            "area_ratio": float((x2 - x1) * (y2 - y1) / img_area),
            "gradient_energy": mean_grad
        })

    # Sort candidates by gradient energy descending (sharpest/heaviest disturbances first)
    candidates.sort(key=lambda item: item["gradient_energy"], reverse=True)
    return candidates[:max_regions]
