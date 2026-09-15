# End-to-End System Architecture & Technical Explanation
## Classical Computer Vision & Tabular ML Vehicle Damage Assessment Pipeline

This document provides a comprehensive, component-by-component explanation of the **100% Classical Computer Vision and Tabular Machine Learning Pipeline** for automated vehicle damage inspection, severity prediction, fraud auditing, and actuarial repair cost estimation.

---

## 1. High-Level System Data Flow

```
                       ┌────────────────────────────────────────┐
                       │           USER UPLOADS IMAGE           │
                       │           (Streamlit: app.py)          │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                   CLASSICAL PIPELINE (src/pipeline.py: assess_image)                     │
│                                                                                          │
│  STAGE 0: Vehicle Isolation (Otsu thresholding + contour centrality)                    │
│           ↳ Suppresses background vehicles & street clutter                              │
│                                                                                          │
│  STAGE 1: Classical Region Proposals (Sobel gradient disturbances + Canny)               │
│           ↳ Extracts candidate damage bounding boxes without YOLO                        │
│                                                                                          │
│  STAGE 2: Domain-Specific Physics Filters (Micro-Analyzers)                              │
│           ↳ Tire Circularity Filter: Measures 4πA/P² to reject round tires               │
│           ↳ Glass Integrity Analyzer: Omnidirectional entropy + specular glare masking   │
│                                                                                          │
│  STAGE 3: 119-Dimensional Classical Feature Extraction                                   │
│           ↳ Color (48) + Texture/GLCM/Gabor (48) + Shape/2D-FFT/Glare (23)              │
│                                                                                          │
│  STAGE 4: Tabular Machine Learning Prediction                                            │
│           ↳ Classifies Crash Severity (Normal / Moderate / Severe) via Classical ML      │
│                                                                                          │
│  STAGE 5: Claim Fraud & Physical Anomaly Auditing                                        │
│           ↳ Checks cross-consistency between localized damages & overall severity       │
│                                                                                          │
│  STAGE 6: Actuarial Cost Engine & Monte Carlo Simulation                                 │
│           ↳ Segment-specific labor/parts pricing + 90% confidence bounds                 │
└──────────────────────────────────────────┬───────────────────────────────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │     INTERACTIVE DASHBOARD RENDERING    │
                       │   - Bounding Boxes & Heatmaps          │
                       │   - 119-Dim Feature Vector Table       │
                       │   - Repair Cost Summary & PDF Export   │
                       └────────────────────────────────────────┘
```

---

## 2. Web Application & Adjuster Dashboard (`app.py`)

The user-facing dashboard is constructed with **Streamlit** to facilitate live claim adjudication by insurance adjusters, underwriters, and researchers:

1. **Sidebar Configuration**:
   - **Vehicle Profile Manager**: Configures vehicle parameters including Make, Model, Year, Segment (*Sedan, SUV, Luxury, Truck*), Actual Cash Value (ACV), and Deductible.
   - **Detection Tuning Toggles**:
     - *Background Vehicle Filter* (`filter_bg`): Toggles vehicle RoI isolation.
     - *Tire Validation Filter* (`validate_tires`): Enables geometric circularity checking.
     - *Glass Shatter Sensitivity* (`glass_sensitivity`): Adjusts edge entropy threshold for windshields and windows.
2. **Image Ingestion**:
   - Accepts real-time file uploads (`.jpg`, `.jpeg`, `.png`) or pre-configured test presets from held-out validation splits.
3. **Interactive Visualizations**:
   - **Annotated Vehicle Canvas**: Draws color-coded damage bounding boxes with associated confidence metrics.
   - **Adjuster Override Tools**: Allows human adjusters to manually insert, modify, or delete damage instances in real-time.
   - **119-Dimension Feature Vector Inspector**: An expandable table showing every physical CV metric extracted from the live image, complete with individual descriptions and a one-click CSV export button.
   - **Financial Adjudication Card**: Itemizes paint hours, body labor, replacement parts, total loss thresholding, and PDF claim report export.

---

## 3. The Core Classical Pipeline (`src/pipeline.py`)

The pipeline operates deterministically in 7 modular stages:

### Stage 0: Subject Vehicle RoI Isolation (`locate_primary_subject_vehicle`)
* **Objective**: Remove irrelevant background elements (parked cars, trees, street signs, pavement).
* **Classical Mechanism**:
  1. Computes Otsu's adaptive thresholding on grayscale luminance to separate foreground objects from the scene background.
  2. Identifies external contours and calculates their geometric centroids:
     $$x_c = \frac{M_{10}}{M_{00}}, \quad y_c = \frac{M_{01}}{M_{00}}$$
  3. Evaluates contour centrality and bounding area. The largest, centrally positioned contour is designated as the primary subject vehicle RoI (`[vx1, vy1, vx2, vy2]`).
  4. Any candidate damage detected outside these spatial coordinates is automatically discarded.

---

### Stage 1: Classical Region Proposals (`propose_damage_regions`)
* **Objective**: Detect damage locations without a deep neural object detector (like YOLOv8).
* **Classical Mechanism**:
  1. Evaluates horizontal and vertical image gradients using Sobel kernels:
     $$G_x = I * K_x, \quad G_y = I * K_y, \quad \text{Magnitude} = \sqrt{G_x^2 + G_y^2}$$
  2. Smooth vehicle body panels produce low gradient energy, while scratches, dents, and fractures create sharp localized gradient disturbances.
  3. Applies morphological dilation and contour clustering to group contiguous edge disturbances into candidate damage bounding boxes.

---

### Stage 2: Physics-Based Micro-Analyzers

To eliminate false positives common in classical methods, dedicated physical filters evaluate candidate regions:

#### 1. Tire Circularity Validator (`validate_tire_flat`)
* Evaluates wheel contours using the **Isoperimetric Quotient (Circularity)**:
  $$\text{Circularity} = 4\pi \frac{\text{Area}}{\text{Perimeter}^2}$$
* Standard inflated tires maintain a round elliptical profile with circularity $> 0.45$. If a detected "tire flat" region retains round geometry and normal aspect ratio, it is suppressed as a false alarm.

#### 2. Glass & Window Integrity Diagnostic Engine (`src/glass_analyzer.py`)
* Distinguishes **sun glare/specular reflection** from **genuine fractured glass**:
  * **Specular Glare Masking**: Isolates high-value, low-saturation pixels in HSV space ($V > 240, S < 30$).
  * **Omnidirectional Edge Entropy**: Glare reflections exhibit smooth, uniform gradients; shattered glass exhibits chaotic, high-entropy micro-fractures across multiple angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
  * Produces a visual **Fracture Density Heatmap**.

---

### Stage 3: Handcrafted 119-Dimensional Feature Extraction (`src/features/`)

The pipeline transforms the vehicle image into an auditable, 119-dimensional tabular feature vector:

| Feature Group | Features | Mathematical Methods & Physical Representation |
| :--- | :---: | :--- |
| **Color Features** | **48** | • **RGB & HSV Channel Moments**: Mean, Standard Deviation, Skewness, Kurtosis (quantifies color degradation, paint fading, exposed underbody).<br>• **8-Bin Color Histograms**: Quantifies hue distribution across channels.<br>• **Color Shannon Entropy**: Measures color complexity.<br>• **$k$-Means Dominant Colors**: Identifies base vehicle coat vs. metallic primer. |
| **Texture Features** | **48** | • **GLCM (Gray-Level Co-occurrence Matrix)**: Evaluates Contrast, Dissimilarity, Homogeneity, Energy, and Correlation across 4 orientations ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).<br>• **Local Binary Patterns (LBP)**: 10-bin histogram capturing micro-surface roughness.<br>• **Multi-Scale Gabor Filter Banks**: 4 frequencies $\times$ 4 orientations measuring directional deformation patterns. |
| **Shape, Frequency & Glare** | **23** | • **Canny Edge Densities**: Fine $(50, 150)$ and Coarse $(100, 200)$ edge-to-area ratios.<br>• **Contour Morphometry**: Aspect ratio, extent, circularity, solidity, convexity defects.<br>• **2D Fast Fourier Transform (FFT)**: High-frequency to low-frequency spectral power ratio (crushed structural elements generate high-frequency Fourier energy).<br>• **Specular Glare Ratio**: Surface area occupied by intense light reflections. |

---

### Stage 4: Tabular Machine Learning Severity Prediction
* Standardizes the 119 features using a fitted `StandardScaler`.
* Feeds tabular data into trained classical classifiers (Logistic Regression, Random Forest, XGBoost).
* Outputs categorical crash severity:
  * `normal`
  * `moderate_breakage`
  * `severe_crushed`
* Generates exact class probability distributions (e.g. `{normal: 0.05, moderate_breakage: 0.15, severe_crushed: 0.80}`).

---

### Stage 5: Claim Fraud & Physical Inconsistency Auditing (`_audit_claim_fraud`)
* Applies actuarial rules to identify anomalous or fraudulent insurance claims:
  * **Inconsistency Check**: Flagged if high crash severity (`severe_crushed`) is predicted with 0 localized damage regions.
  * **Under-classification Check**: Flagged if crash is classified as `normal` despite $\ge 4$ localized damage detections.
  * **Frame Area Over-coverage**: Flagged if damage spans $>60\%$ of the image frame (indicative of multi-vehicle pileups or camera artifacts).
* Assigns an **Anomaly Score (0.0 to 1.0)** and risk tiers: `LOW`, `MEDIUM`, or `HIGH`.

---

### Stage 6: Actuarial Cost Engine & Monte Carlo Simulation (`src/cost_estimator.py`)
1. **Base Damage Cost Matrix**:
   - Assigns baseline labor hours (bodywork, paint) and OEM part replacement costs to each damage category (*dent, scratch, crack, glass shatter, lamp broken, tire flat*).
2. **Segment Multipliers**:
   - Scales costs according to vehicle classification:
     - *Luxury / Exotic*: $1.60\times$ parts multiplier, higher labor rates.
     - *Full-size SUV / Truck*: $1.20\times$ multiplier.
     - *Midsize Sedan*: $1.00\times$ baseline.
     - *Compact / Economy*: $0.85\times$ multiplier.
3. **Monte Carlo Stochastic Simulation**:
   - Executes 1,000 randomized iterations sampling from normal distributions of labor hours and supply chain part volatility.
   - Computes a **90% Confidence Interval** for total repair cost.
4. **Total Loss Determination**:
   - Compares total estimated repair cost against the **Total Loss Threshold** ($75\%$ of vehicle Actual Cash Value).
   - Flags the claim as **Total Loss (Write-Off)** if repairs exceed the financial threshold.

---

## 4. Academic & Research Presentation Summary

| Attribute | Deep Learning Pipeline (Legacy) | Classical ML Pipeline (Current) |
| :--- | :--- | :--- |
| **Feature Extraction** | Black-box latent embeddings (ResNet50 / YOLO) | 119 Handcrafted Physical Metrics (Color, GLCM, LBP, Gabor, FFT) |
| **Interpretability** | Low (Opaque activations) | **100% Inspectable** via Excel/CSV feature tables |
| **Auditability** | Difficult for claims adjusters | Direct mathematical mapping to physical phenomena |
| **Computational Footprint** | Requires GPU hardware | **Lightweight CPU-Native Execution** |
| **Domain Control** | Probabilistic network output | Deterministic optical filters (Tire circularity & Glare entropy) |
