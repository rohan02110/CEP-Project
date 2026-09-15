"""
Classical Computer Vision Feature Extraction Package for Vehicle Damage Assessment.
This package implements interpretable, handcrafted feature engineering:
- Preprocessing & classical region segmentation
- Color space histograms and statistical moments
- Texture analysis (GLCM, LBP, Gabor filter banks)
- Shape, contour topology, circularity, 2D FFT frequency energy, and specular reflectance
"""

from src.features.preprocessing import preprocess_image, isolate_vehicle_roi, propose_damage_regions
from src.features.color_features import extract_color_features, get_color_feature_names
from src.features.texture_features import extract_texture_features, get_texture_feature_names
from src.features.shape_features import extract_shape_features, get_shape_feature_names
from src.features.build_feature_table import extract_all_features_from_image, build_feature_dataset

__all__ = [
    "preprocess_image",
    "isolate_vehicle_roi",
    "propose_damage_regions",
    "extract_color_features",
    "get_color_feature_names",
    "extract_texture_features",
    "get_texture_feature_names",
    "extract_shape_features",
    "get_shape_feature_names",
    "extract_all_features_from_image",
    "build_feature_dataset"
]
