"""
Unit Tests for Classical Computer Vision Feature Extraction Modules.
Verifies dimensionality, fixed-length guarantees, zero NaN/Inf assertions,
edge cases (black, white, monochrome, noisy images), and bounded metrics.
"""

import sys
import unittest
from pathlib import Path

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import numpy as np
from PIL import Image

from src.features.preprocessing import preprocess_image, isolate_vehicle_roi, propose_damage_regions
from src.features.color_features import extract_color_features, get_color_feature_names
from src.features.texture_features import extract_texture_features, get_texture_feature_names
from src.features.shape_features import extract_shape_features, get_shape_feature_names
from src.features.build_feature_table import extract_all_features_from_image, get_full_feature_dictionary


class TestClassicalFeatureExtraction(unittest.TestCase):

    def setUp(self):
        """Creates standard synthetic test images for unit testing."""
        np.random.seed(42)
        
        # 1. Realistic synthetic car panel (blue metallic with a bright scratch and shadow)
        self.sample_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        self.sample_rgb[:, :] = [30, 80, 180]  # Base metallic blue
        # Add white diagonal scratch
        cv2.line(self.sample_rgb, (30, 40), (220, 210), (240, 240, 250), 3)
        # Add dark dent shadow
        cv2.circle(self.sample_rgb, (150, 120), 40, (15, 40, 90), -1)

        # 2. Perfect circle image (for circularity verification)
        self.circle_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(self.circle_img, (100, 100), 60, (255, 255, 255), -1)

        # 3. Flat black and flat white edge cases
        self.black_img = np.zeros((128, 128, 3), dtype=np.uint8)
        self.white_img = np.full((128, 128, 3), 255, dtype=np.uint8)

        # 4. Random noise image
        self.noise_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    def test_color_feature_extraction(self):
        """Tests color feature dimensionality, zero NaNs, and bounds."""
        names = get_color_feature_names()
        self.assertEqual(len(names), 48, f"Expected 48 color features, got {len(names)}")

        for img in [self.sample_rgb, self.black_img, self.white_img, self.noise_img]:
            feats = extract_color_features(img)
            self.assertEqual(len(feats), 48)
            
            # Check all values are finite and not NaN
            for k, v in feats.items():
                self.assertIsInstance(v, float)
                self.assertFalse(np.isnan(v), f"NaN detected in color feature: {k}")
                self.assertFalse(np.isinf(v), f"Inf detected in color feature: {k}")

            # Verify RGB/HSV histogram sum is approx 1.0 (or 0 for empty)
            r_hist_sum = sum(feats[f"color_hist_r_bin{b}"] for b in range(4))
            self.assertAlmostEqual(r_hist_sum, 1.0, places=2)

    def test_texture_feature_extraction(self):
        """Tests GLCM, LBP, and Gabor texture feature extraction."""
        names = get_texture_feature_names()
        self.assertEqual(len(names), 48, f"Expected 48 texture features, got {len(names)}")

        for img in [self.sample_rgb, self.black_img, self.white_img, self.noise_img]:
            feats = extract_texture_features(img)
            self.assertEqual(len(feats), 48)
            
            for k, v in feats.items():
                self.assertIsInstance(v, float)
                self.assertFalse(np.isnan(v), f"NaN detected in texture feature: {k}")
                self.assertFalse(np.isinf(v), f"Inf detected in texture feature: {k}")

            # Entropy bounds
            self.assertGreaterEqual(feats["texture_lbp_entropy"], 0.0)
            self.assertLessEqual(feats["texture_lbp_entropy"], 1.0)

    def test_shape_and_frequency_features(self):
        """Tests shape, circularity, 2D FFT, and glare extraction."""
        names = get_shape_feature_names()
        self.assertEqual(len(names), 23, f"Expected 23 shape features, got {len(names)}")

        for img in [self.sample_rgb, self.black_img, self.white_img, self.noise_img]:
            feats = extract_shape_features(img)
            self.assertEqual(len(feats), 23)
            
            for k, v in feats.items():
                self.assertIsInstance(v, float)
                self.assertFalse(np.isnan(v), f"NaN detected in shape feature: {k}")
                self.assertFalse(np.isinf(v), f"Inf detected in shape feature: {k}")

            # Check bounded metric invariants
            self.assertGreaterEqual(feats["shape_circularity_primary"], 0.0)
            self.assertLessEqual(feats["shape_circularity_primary"], 1.0)
            self.assertGreaterEqual(feats["shape_specular_glare_ratio"], 0.0)
            self.assertLessEqual(feats["shape_specular_glare_ratio"], 1.0)
            self.assertGreaterEqual(feats["shape_solidity_primary"], 0.0)
            self.assertLessEqual(feats["shape_solidity_primary"], 1.0)

        # Test circularity detection logic specifically
        circle_feats = extract_shape_features(self.circle_img)
        self.assertGreater(
            circle_feats["shape_circularity_primary"], 0.70,
            f"Expected circle circularity > 0.70, got {circle_feats['shape_circularity_primary']}"
        )

    def test_preprocessing_and_proposals(self):
        """Tests image preprocessing, vehicle ROI isolation, and damage proposals."""
        clean = preprocess_image(self.sample_rgb, target_size=(256, 256), denoise_method="bilateral")
        self.assertEqual(clean.shape, (256, 256, 3))
        self.assertEqual(clean.dtype, np.uint8)

        roi_crop, bbox = isolate_vehicle_roi(self.sample_rgb)
        self.assertGreater(roi_crop.size, 0)
        self.assertEqual(len(bbox), 4)
        self.assertLessEqual(bbox[0], bbox[2])
        self.assertLessEqual(bbox[1], bbox[3])

        proposals = propose_damage_regions(self.sample_rgb, max_regions=3)
        self.assertIsInstance(proposals, list)
        if len(proposals) > 0:
            self.assertIn("bbox", proposals[0])
            self.assertIn("crop", proposals[0])
            self.assertIn("gradient_energy", proposals[0])

    def test_full_pipeline_consistency(self):
        """Tests end-to-end extract_all_features_from_image and dictionary alignment."""
        full_dict = extract_all_features_from_image(self.sample_rgb)
        # Expected total: 48 (color) + 48 (texture) + 23 (shape) = 119 features
        self.assertEqual(len(full_dict), 119, f"Expected 119 total features, got {len(full_dict)}")

        meta_df = get_full_feature_dictionary()
        self.assertEqual(len(meta_df), 119)
        self.assertEqual(set(full_dict.keys()), set(meta_df["Feature_Name"]))


if __name__ == "__main__":
    unittest.main()
