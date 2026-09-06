# AI-Assisted Vehicle Damage Assessment & Preliminary Insurance Claim Estimation 🚗⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision and machine learning framework designed to automate preliminary motor insurance claim processing. The system combines multi-class localized damage detection (**YOLOv8**), deep visual feature representation (**ResNet50**), multi-class severity classification, and a knowledge-grounded repair-cost estimation engine.

Developed for academic presentation at the **Young Researcher Conference**.

---

## 📌 Abstract & Problem Statement

Motor insurance claims traditionally require manual on-site inspection, taking days to process document verification, fraud checks, and preliminary repair cost estimation. 

This project delivers an automated computer-vision pipeline that:
1. **Detects & Localizes 6 Damage Types** (`scratch`, `dent`, `crack`, `glass shatter`, `lamp broken`, `tire flat`) using a fine-tuned **YOLOv8** object detector.
2. **Computes 3-Level Severity** (`normal`, `moderate_breakage`, `severe_crushed`) using deep CNN embeddings ($2048$-dim) and multi-class classifiers.
3. **Generates Defensible Cost Ranges** ($\hat{Y}_{\text{cost\_low}} \rightarrow \hat{Y}_{\text{cost\_high}}$) with itemized component replacement and labor hour multipliers.
4. **Ensures Explainability** via Grad-CAM activation heatmaps and interactive inspection dashboards.

---

## 📊 Experimental Results & Benchmark Performance

### **1. YOLOv8 Damage Detector (CarDD Test Set, $N=120$ Unseen Images, $256$ Instances)**
- **mAP@50**: **`54.24%`**
- **mAP@50:95**: **`40.68%`**
- **Precision**: **`49.07%`**
- **Recall**: **`54.42%`**
- **Inference Latency**: **`19.4 ms`** ($\sim 51.5$ FPS real-time execution)

#### **Per-Class Test Performance (mAP@50:95)**
| Damage Category | Test Instances | Precision | Recall | mAP@50 | mAP@50:95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🛞 **tire flat** | 22 | **72.2%** | **86.4%** | **90.9%** | **87.03%** |
| 🪟 **glass shatter** | 28 | **70.7%** | **94.8%** | **93.9%** | **79.16%** |
| 💡 **lamp broken** | 21 | **45.2%** | **71.4%** | **65.6%** | **39.55%** |
| 🖌️ **scratch** | 83 | **41.9%** | **30.1%** | **33.2%** | **18.30%** |
| 💥 **dent** | 84 | **42.6%** | **32.7%** | **32.8%** | **15.68%** |
| ⚡ **crack** | 18 | **21.8%** | **11.1%** | **9.1%** | **4.35%** |

---

## 🛡️ Research Integrity & Zero Data-Leakage Policy
To prevent data contamination, all datasets were partitioned using strict cryptographic MD5 hash disjointness checks prior to training:
$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- **CarDD Dataset**: 560 Train ($1,181$ bboxes), 120 Val ($272$ bboxes), 120 Test ($256$ bboxes).
- **Severity Benchmark**: 420 Train ($140$/class), 90 Val ($30$/class), 90 Test ($30$/class).

---

## 🛠️ Repository Structure

```
CEP-Project/
├── data/                  # Auto-created directory structure
├── experiments/
│   ├── models/            # Saved model checkpoints (yolov8_damage_best.pt)
│   ├── plots/             # Publication-grade figures & prediction grids
│   └── metrics/           # Machine-readable JSON evaluation metrics
├── src/
│   ├── config.py          # Centralized configuration, seeds & paths
│   ├── data_loader_eda.py # Multi-threaded dataset downloader & EDA
│   ├── preprocess.py      # YOLO formatting & zero-leakage split engine
│   └── train_detector.py  # YOLOv8 fine-tuning & test set evaluator
├── requirements.txt       # Python dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/rohan02110/CEP-Project.git
cd CEP-Project
pip install -r requirements.txt
```

### 2. Download Datasets & Run Exploratory Data Analysis (EDA)
```bash
python src/data_loader_eda.py
```

### 3. Preprocess & Partition Datasets (Zero-Leakage Guarantee)
```bash
python src/preprocess.py
```

### 4. Fine-Tune & Evaluate YOLOv8 Damage Detector
```bash
python src/train_detector.py
```

---

## ⚖️ Academic License
Distributed under the MIT License. Developed for undergraduate academic research.
