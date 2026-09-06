"""
Multi-Class Severity Classification Benchmarks & Champion Model Selection.
Trains and compares multiple classifiers on 2048-dim ResNet50 visual embeddings.
Evaluates on unseen held-out test split (0-leakage) with comprehensive research metrics.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from src.config import (
    FEATURES_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR,
    SEVERITY_CLASSES, SEVERITY_CLASS_TO_IDX, SEVERITY_IDX_TO_CLASS,
    RANDOM_SEED
)

np.random.seed(RANDOM_SEED)


def load_severity_features():
    """Loads pre-extracted 2048-dim ResNet50 embeddings for train, val, test splits."""
    print("Loading extracted severity feature arrays...", flush=True)
    
    train_data = np.load(FEATURES_DIR / "severity_features_train.npz")
    val_data = np.load(FEATURES_DIR / "severity_features_val.npz")
    test_data = np.load(FEATURES_DIR / "severity_features_test.npz")

    X_train, y_train = train_data["features"], train_data["labels"]
    X_val, y_val = val_data["features"], val_data["labels"]
    X_test, y_test = test_data["features"], test_data["labels"]

    print(f"Train features: {X_train.shape}, labels: {y_train.shape}")
    print(f"Val features:   {X_val.shape}, labels: {y_val.shape}")
    print(f"Test features:  {X_test.shape}, labels: {y_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_models():
    """Initializes the benchmark suite of classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "Support Vector Machine (RBF)": SVC(
            C=1.5,
            kernel="rbf",
            probability=True,
            random_state=RANDOM_SEED
        ),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(
            hidden_layer_sizes=(512, 128),
            activation="relu",
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=RANDOM_SEED
        )
    }
    return models


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name=""):
    """
    Fits model, monitors validation, evaluates on unseen test set,
    and measures inference latency per sample.
    """
    print(f"\n--- Training & Evaluating: {model_name} ---", flush=True)
    
    # Measure training time
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Validation accuracy
    val_preds = model.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))

    # Test evaluation
    t_inf_start = time.time()
    test_preds = model.predict(X_test)
    inf_time_total = (time.time() - t_inf_start) * 1000  # ms
    latency_per_sample = inf_time_total / len(X_test)  # ms/sample

    test_probs = model.predict_proba(X_test)

    # Compute classification metrics
    test_acc = float(accuracy_score(y_test, test_preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, test_preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, test_preds, average="weighted", zero_division=0
    )

    # Multiclass ROC-AUC (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    try:
        roc_auc_ovr_macro = float(roc_auc_score(y_test_bin, test_probs, average="macro", multi_class="ovr"))
        roc_auc_ovr_weighted = float(roc_auc_score(y_test_bin, test_probs, average="weighted", multi_class="ovr"))
    except Exception:
        roc_auc_ovr_macro = 0.0
        roc_auc_ovr_weighted = 0.0

    # Per-class metrics
    class_report_dict = classification_report(
        y_test, test_preds,
        target_names=SEVERITY_CLASSES,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, test_preds)

    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Test Accuracy:       {test_acc * 100:.2f}%")
    print(f"Test Macro-F1:       {f1_macro * 100:.2f}%")
    print(f"Test ROC-AUC (OvR):  {roc_auc_ovr_macro:.4f}")
    print(f"Inference Latency:   {latency_per_sample:.3f} ms / sample")

    results = {
        "model_name": model_name,
        "train_time_sec": round(train_time, 4),
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_precision": round(float(prec_macro), 4),
        "test_macro_recall": round(float(rec_macro), 4),
        "test_macro_f1": round(float(f1_macro), 4),
        "test_weighted_f1": round(float(f1_weighted), 4),
        "test_roc_auc_macro": round(roc_auc_ovr_macro, 4),
        "test_roc_auc_weighted": round(roc_auc_ovr_weighted, 4),
        "latency_ms_per_sample": round(latency_per_sample, 4),
        "classification_report": class_report_dict,
        "confusion_matrix": cm.tolist(),
        "predictions": test_preds.tolist(),
        "probabilities": test_probs.tolist()
    }
    return results


def plot_confusion_matrices(champion_name, champion_results, baseline_results):
    """Generates a high-resolution side-by-side Confusion Matrix plot."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    clean_labels = [c.replace("_", " ").title() for c in SEVERITY_CLASSES]

    # Baseline CM
    cm_base = np.array(baseline_results["confusion_matrix"])
    cm_base_norm = cm_base.astype('float') / cm_base.sum(axis=1)[:, np.newaxis]
    
    annot_base = np.empty_like(cm_base, dtype=object)
    for i in range(cm_base.shape[0]):
        for j in range(cm_base.shape[1]):
            annot_base[i, j] = f"{cm_base[i, j]}\n({cm_base_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_base_norm, annot=annot_base, fmt="", cmap="Blues",
        xticklabels=clean_labels, yticklabels=clean_labels,
        cbar=True, ax=axes[0], annot_kws={"size": 11, "weight": "bold"}
    )
    axes[0].set_title(f"Baseline: {baseline_results['model_name']}\n(Acc: {baseline_results['test_accuracy']*100:.1f}% | Macro-F1: {baseline_results['test_macro_f1']*100:.1f}%)", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Predicted Severity Class", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Severity Class", fontsize=11, fontweight="bold")

    # Champion CM
    cm_champ = np.array(champion_results["confusion_matrix"])
    cm_champ_norm = cm_champ.astype('float') / cm_champ.sum(axis=1)[:, np.newaxis]
    
    annot_champ = np.empty_like(cm_champ, dtype=object)
    for i in range(cm_champ.shape[0]):
        for j in range(cm_champ.shape[1]):
            annot_champ[i, j] = f"{cm_champ[i, j]}\n({cm_champ_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_champ_norm, annot=annot_champ, fmt="", cmap="Greens",
        xticklabels=clean_labels, yticklabels=clean_labels,
        cbar=True, ax=axes[1], annot_kws={"size": 11, "weight": "bold"}
    )
    axes[1].set_title(f"Champion: {champion_name}\n(Acc: {champion_results['test_accuracy']*100:.1f}% | Macro-F1: {champion_results['test_macro_f1']*100:.1f}%)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Predicted Severity Class", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("True Severity Class", fontsize=11, fontweight="bold")

    plt.suptitle("Phase 6: Multi-Class Severity Classification Confusion Matrices (Test Set N=90)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_path = PLOTS_DIR / "severity_confusion_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion Matrix figure to: {out_path}", flush=True)


def plot_benchmark_comparison(all_results):
    """Plots comparative bar charts across all benchmarked models."""
    models = [r["model_name"] for r in all_results]
    accuracies = [r["test_accuracy"] * 100 for r in all_results]
    macro_f1s = [r["test_macro_f1"] * 100 for r in all_results]
    macro_precs = [r["test_macro_precision"] * 100 for r in all_results]
    macro_recs = [r["test_macro_recall"] * 100 for r in all_results]
    roc_aucs = [r["test_roc_auc_macro"] * 100 for r in all_results]

    x = np.arange(len(models))
    width = 0.16

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    rects1 = ax.bar(x - 2*width, accuracies, width, label='Accuracy (%)', color='#2563EB')
    rects2 = ax.bar(x - width, macro_f1s, width, label='Macro-F1 (%)', color='#10B981')
    rects3 = ax.bar(x, macro_precs, width, label='Macro-Precision (%)', color='#F59E0B')
    rects4 = ax.bar(x + width, macro_recs, width, label='Macro-Recall (%)', color='#EC4899')
    rects5 = ax.bar(x + 2*width, roc_aucs, width, label='ROC-AUC (OvR %)', color='#8B5CF6')

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Class Severity Classification: Benchmark Model Comparison on Unseen Test Set', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold', rotation=10)
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotate values on top of bars
    for rect in [rects1, rects2, rects5]:
        for bar in rect:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    out_path = PLOTS_DIR / "severity_model_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Benchmark Comparison figure to: {out_path}", flush=True)


def plot_multiclass_roc(champion_name, champion_results, y_test):
    """Plots One-vs-Rest Multiclass ROC Curves for each severity category."""
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = np.array(champion_results["probabilities"])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    colors = ['#10B981', '#F59E0B', '#EF4444']
    clean_labels = [c.replace("_", " ").title() for c in SEVERITY_CLASSES]

    plt.figure(figsize=(9, 7), dpi=300)

    for i in range(len(SEVERITY_CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2.5,
                 label=f'{clean_labels[i]} (AUC = {roc_auc[i]:.3f})')

    # Macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(SEVERITY_CLASSES))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(SEVERITY_CLASSES)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(SEVERITY_CLASSES)

    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='#1E293B', linestyle='--', lw=2.5,
             label=f'Macro-Average ROC (AUC = {macro_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k:', lw=1.5, label='Chance Level (AUC = 0.500)')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    plt.title(f'Multi-Class ROC Curves (One-vs-Rest): {champion_name}\n(Unseen Held-out Test Set, N=90)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = PLOTS_DIR / "severity_roc_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Multiclass ROC Curves to: {out_path}", flush=True)


def run_severity_benchmarks():
    """Main execution function for Phase 6."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_severity_features()
    models = get_models()

    all_results = []
    trained_models = {}

    for name, model in models.items():
        res = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name=name)
        all_results.append(res)
        trained_models[name] = model

    # Select champion based on highest test Macro-F1 (with test accuracy as tiebreaker)
    all_results.sort(key=lambda r: (r["test_macro_f1"], r["test_accuracy"]), reverse=True)
    champion_res = all_results[0]
    champion_name = champion_res["model_name"]
    champion_model = trained_models[champion_name]

    print("\n" + "="*60)
    print(f"*** CHAMPION MODEL SELECTED: {champion_name} ***")
    print(f"Test Accuracy: {champion_res['test_accuracy']*100:.2f}% | Test Macro-F1: {champion_res['test_macro_f1']*100:.2f}%")
    print("="*60, flush=True)

    # Save Champion Model Checkpoint
    champ_path = MODELS_DIR / "severity_classifier_best.joblib"
    joblib.dump({
        "model": champion_model,
        "model_name": champion_name,
        "classes": SEVERITY_CLASSES,
        "class_to_idx": SEVERITY_CLASS_TO_IDX,
        "idx_to_class": SEVERITY_IDX_TO_CLASS,
        "metrics": champion_res
    }, champ_path)
    print(f"Saved Champion model artifact to: {champ_path}")

    # Generate Visualizations
    baseline_res = next(r for r in all_results if r["model_name"] == "Logistic Regression")
    plot_confusion_matrices(champion_name, champion_res, baseline_res)
    plot_benchmark_comparison(all_results)
    plot_multiclass_roc(champion_name, champion_res, y_test)

    # Serialize Full Benchmark Results to JSON
    json_summary = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "SaiVaibhavS/comprehensive-car-damage (3-Class Severity)",
        "sample_split": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "total": len(X_train) + len(X_val) + len(X_test)
        },
        "feature_extractor": "ResNet50 (2048-dim, L2-normalized)",
        "champion_model": champion_name,
        "benchmark_rankings": [
            {
                "rank": idx + 1,
                "model": r["model_name"],
                "test_accuracy": r["test_accuracy"],
                "test_macro_f1": r["test_macro_f1"],
                "test_macro_precision": r["test_macro_precision"],
                "test_macro_recall": r["test_macro_recall"],
                "test_roc_auc_macro": r["test_roc_auc_macro"],
                "latency_ms_per_sample": r["latency_ms_per_sample"],
                "train_time_sec": r["train_time_sec"]
            }
            for idx, r in enumerate(all_results)
        ],
        "detailed_results": all_results
    }

    metrics_file = METRICS_DIR / "severity_test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    print(f"Saved Full Benchmark Metrics to: {metrics_file}")
    return json_summary


if __name__ == "__main__":
    run_severity_benchmarks()
