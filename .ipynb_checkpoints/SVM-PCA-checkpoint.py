"""
Web Attack Detection: SVM + TF-IDF/PCA Pipeline
=================================================
CSCI 459 - AI Enhanced Security | Final Project

Binary classification of HTTP API requests as Benign or Malicious using:
    TF-IDF (character n-grams) -> TruncatedSVD -> LinearSVC [PCA-reduced SVM]

Feature engineering includes HTTP method, URL, body, AND security-relevant headers
(Cookie, X-Forwarded-For, Referer, User-Agent) which carry strong attack signals for
Cookie Injection, Log Forging, and RCE payloads.

Dataset: ATRDF 2023 (Cisco-Ariel University API Security Challenge)
Reference: Aharon et al., Computers & Security, Elsevier, 2025.

Outputs (all saved to ./outputs/):
    attack_distribution.png   — attack type counts across all 4 datasets
    explained_variance.png    — SVD cumulative variance across all 4 datasets
    c_tuning.png              — val F1 vs C across all 4 datasets
    cm_validation.png         — confusion matrices (validation) all 4 datasets
    cm_test.png               — confusion matrices (test) all 4 datasets
    metrics.png               — accuracy/precision/recall/F1 across datasets
    radar.png                 — radar chart of all metrics per dataset
    roc_curves.png            — ROC curves + AUC across all datasets
    timing.png                — train time & inference latency per dataset

Requirements:
"pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib"
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from urllib.parse import unquote
from scipy.sparse import vstack as spvstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR        = Path("./Datasets")
ALL_DATASETS       = [1, 2, 3, 4]
OUTPUT_DIR         = Path("./outputs")

TFIDF_MAX_FEATURES = 5_000
TFIDF_NGRAM_RANGE  = (3, 5)
N_COMPONENTS       = 50
C_VALUES           = [0.0001, 0.01, 0.1, 10.0]
RANDOM_STATE       = 35
TRAIN_FRAC         = 0.70
VAL_FRAC           = 0.15

SECURITY_HEADERS = [
    "Cookie",
    "X-Forwarded-For",
    "Referer",
    "User-Agent",
    "Authorization",
    "X-Api-Key",
]

COLORS = ["steelblue", "darkorange", "seagreen", "mediumpurple"]


# =============================================================================
# HELPERS
# =============================================================================

def loadAndFlatten(path):
    """Load a dataset JSON and flatten the nested request object."""
    raw = pd.read_json(path)
    df  = pd.json_normalize(raw["request"])
    if "Attack_Tag" in df.columns:
        df["label"]       = df["Attack_Tag"].apply(
            lambda x: "Malicious" if pd.notna(x) and str(x).strip() != "" else "Benign"
        )
        df["attack_type"] = df["Attack_Tag"].fillna("Benign")
    else:
        df["label"]       = "Benign"
        df["attack_type"] = "Benign"
    df["_headers_raw"] = raw["request"].apply(
        lambda r: r.get("headers", {}) if isinstance(r, dict) else {}
    )
    return df


def buildText(row):
    """
    Concatenate HTTP method, URL, body, and security-relevant header values
    into a single string for TF-IDF vectorisation.

    Security headers (Cookie, X-Forwarded-For, Referer, User-Agent, etc.) are
    appended because they carry strong attack signals:
        - Cookie Injection  -> malicious payloads in the Cookie header
        - Log Forging       -> CRLF sequences in User-Agent / Referer
        - RCE               -> serialised objects in Cookie / Authorization
    """
    parts = []

    for col in ["method", "url", "body"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val))

    headers = row.get("_headers_raw", {})
    if isinstance(headers, dict):
        for hname in SECURITY_HEADERS:
            hval = headers.get(hname, "")
            if hval and str(hval).strip():
                parts.append(f"{hname.lower()} {str(hval)}")

    text = " ".join(parts)
    try:
        text = unquote(text)
    except Exception:
        pass
    return text.lower().strip()


# =============================================================================
# PLOTS
# =============================================================================

def plotAttackDistribution(allDfs):
    """Grouped bar chart of attack type counts across all datasets."""
    allTypes = set()
    for df in allDfs.values():
        allTypes.update(df["attack_type"].unique())
    allTypes = sorted(allTypes)

    # Append total malicious as the final column group
    displayTypes = allTypes + ["Total Malicious"]

    x     = np.arange(len(displayTypes))
    n     = len(allDfs)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (dsNum, df) in enumerate(allDfs.items()):
        counts     = df["attack_type"].value_counts()
        vals       = [counts.get(t, 0) for t in allTypes]
        nMalicious = int((df["label"] == "Malicious").sum())
        vals       = vals + [nMalicious]
        offset     = (i - n / 2 + 0.5) * width
        bars       = ax.bar(x + offset, vals, width,
                            label=f"Dataset {dsNum} ({nMalicious:,} malicious)",
                            color=COLORS[i], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        str(v), ha="center", va="bottom", fontsize=7)

    # Vertical line to visually separate the total column
    ax.axvline(x=len(allTypes) - 0.5, color="black", linestyle="--",
               linewidth=1, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(displayTypes, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(
        "Attack Type Distribution Across All Datasets\n"
        "(All non-Benign types are collapsed into 'Malicious' for classification)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attack_distribution.png", dpi=120)
    plt.close()


def plotExplainedVariance(allSvds):
    """Cumulative explained variance curves for all datasets on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (dsNum, svd) in enumerate(allSvds.items()):
        cumVar = np.cumsum(svd.explained_variance_ratio_)
        ax.plot(range(1, len(cumVar) + 1), cumVar,
                color=COLORS[i], linewidth=2, label=f"Dataset {dsNum}")

    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1, label="95% threshold")
    ax.set_xlabel("Number of SVD Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"TruncatedSVD — Cumulative Explained Variance (All Datasets)\n"
                 f"(Using {N_COMPONENTS} components)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "explained_variance.png", dpi=120)
    plt.close()


def plotCTuning(allTuning):
    """Line plot of val F1 vs C for all datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (dsNum, tDict) in enumerate(allTuning.items()):
        tuningDf = tDict["pca"]
        ax.plot([str(c) for c in tuningDf["C"]], tuningDf["val_f1"],
                marker="o", color=COLORS[i], linewidth=2,
                label=f"Dataset {dsNum}")
        bestIdx = tuningDf["val_f1"].idxmax()
        bestC   = tuningDf.loc[bestIdx, "C"]
        bestF1  = tuningDf.loc[bestIdx, "val_f1"]
        ax.scatter([str(bestC)], [bestF1], color=COLORS[i], s=140,
                   zorder=5, marker="*")

    ax.set_xlabel("C (Regularization Parameter)")
    ax.set_ylabel("Macro F1 Score (Validation)")
    ax.set_title("Hyperparameter Tuning — C vs. Validation F1 (PCA-Reduced SVM)\n"
                 "(★ = best C selected per dataset)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "c_tuning.png", dpi=120)
    plt.close()


def plotConfusionMatrices(allCms, labelNames, splitName, savePath):
    """2x2 grid of confusion matrices."""
    n   = len(allCms)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (dsNum, cm) in enumerate(allCms.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=labelNames)
        disp.plot(cmap="Blues", ax=axes[i], colorbar=False)
        axes[i].set_title(f"Dataset {dsNum}", fontsize=12)

    for j in range(n, 4):
        axes[j].set_visible(False)

    fig.suptitle(f"Confusion Matrices — {splitName} Split (PCA-Reduced SVM)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(savePath, dpi=120)
    plt.close()


def plotMetrics(allResults):
    """4-panel bar chart: accuracy/precision/recall/F1 per dataset."""
    datasets    = [f"DS {r['dataset']}" for r in allResults]
    metrics     = ["accuracy", "precision", "recall", "f1"]
    metricNames = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    x     = np.arange(len(datasets))
    width = 0.5

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (metric, name) in enumerate(zip(metrics, metricNames)):
        vals = [r["pca"]["test_metrics"][metric] for r in allResults]
        bars = axes[i].bar(x, vals, width, color=COLORS[:len(vals)], alpha=0.85)

        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                         f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

        axes[i].set_title(name)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(datasets)
        axes[i].set_ylim(max(0, min(vals) - 0.05), 1.05)
        axes[i].set_ylabel("Score")
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle("PCA-Reduced SVM — Test Metrics Across All Datasets",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics.png", dpi=120, bbox_inches="tight")
    plt.close()


def plotROCCurves(allResults):
    """
    ROC curves + AUC across all datasets.

    LinearSVC does not output probabilities; we use decision_function scores
    (signed distance to hyperplane) as the ranking score for ROC computation.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for colIdx, r in enumerate(allResults):
        ax = axes[colIdx]
        fpr, tpr = r["pca"]["roc"]
        aucVal   = r["pca"]["auc"]

        ax.plot(fpr, tpr, color="darkorange", linewidth=2,
                label=f"AUC = {aucVal:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.12, color="darkorange")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(f"Dataset {r['dataset']}", fontsize=10)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("ROC Curves — PCA-Reduced SVM (All Datasets)\n"
                 "Score = LinearSVC decision_function distance to hyperplane",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close()


def plotTiming(allResults):
    """Bar chart of training time and inference latency per dataset."""
    datasets = [f"DS {r['dataset']}" for r in allResults]
    x        = np.arange(len(datasets))
    width    = 0.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    trainTimes = [r["pca"]["test_metrics"]["train_time"] for r in allResults]
    bars1 = ax1.bar(x, trainTimes, width, color=COLORS[:len(trainTimes)], alpha=0.85)
    for bar, v in zip(bars1, trainTimes):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                 f"{v:.2f}s", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Training Time (final model on train+val)")
    ax1.grid(axis="y", alpha=0.3)

    infTimes = [r["pca"]["test_metrics"]["inference_time"] * 1000 for r in allResults]
    bars2 = ax2.bar(x, infTimes, width, color=COLORS[:len(infTimes)], alpha=0.85)
    for bar, v in zip(bars2, infTimes):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                 f"{v:.1f}ms", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_ylabel("Latency (milliseconds)")
    ax2.set_title("Inference Latency on Test Split")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("PCA-Reduced SVM — Training Time & Inference Latency",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "timing.png", dpi=120)
    plt.close()


# =============================================================================
# PER-DATASET PIPELINE
# =============================================================================

def fitOneC(C, XTrain, yTrain, XVal, yVal):
    """Fit one LinearSVC with the given C and return validation metrics."""
    model = svm.LinearSVC(C=C, class_weight="balanced",
                          max_iter=5000, random_state=RANDOM_STATE)
    model.fit(XTrain, yTrain)
    yVPred = model.predict(XVal)
    prec, rec, f1, _ = precision_recall_fscore_support(
        yVal, yVPred, average="macro", zero_division=0)
    acc = accuracy_score(yVal, yVPred)
    return {"C": C, "val_f1": f1, "val_acc": acc,
            "val_prec": prec, "val_rec": rec}


def tuneAndEval(XTrain, yTrain, XVal, yVal, XTest, yTest):
    """
    Run C-grid search on validation set, retrain on train+val, evaluate on test.
    Returns (tuning_df, best_C, val_metrics, val_cm, test_metrics, test_cm,
             roc_tuple, auc_value, final_model).
    """
    tuningRows = joblib.Parallel(n_jobs=len(C_VALUES), prefer="threads")(
        joblib.delayed(fitOneC)(C, XTrain, yTrain, XVal, yVal)
        for C in C_VALUES
    )
    tuningDf = pd.DataFrame(tuningRows)
    bestC    = tuningDf.loc[tuningDf["val_f1"].idxmax(), "C"]

    valModel = svm.LinearSVC(C=bestC, class_weight="balanced",
                             max_iter=5000, random_state=RANDOM_STATE)
    valModel.fit(XTrain, yTrain)
    yVPred = valModel.predict(XVal)
    vAcc = accuracy_score(yVal, yVPred)
    vp, vr, vf, _ = precision_recall_fscore_support(
        yVal, yVPred, average="macro", zero_division=0)
    valMetrics = {"accuracy": vAcc, "precision": vp, "recall": vr, "f1": vf}
    valCm = confusion_matrix(yVal, yVPred)

    XFinal = np.vstack([XTrain, XVal])
    yFinal = np.concatenate([yTrain, yVal])

    finalModel = svm.LinearSVC(C=bestC, class_weight="balanced",
                               max_iter=5000, random_state=RANDOM_STATE)
    t0 = time.time()
    finalModel.fit(XFinal, yFinal)
    trainTime = time.time() - t0

    t0     = time.time()
    yTPred = finalModel.predict(XTest)
    infTime = time.time() - t0
    tAcc = accuracy_score(yTest, yTPred)
    tp, tr, tf, _ = precision_recall_fscore_support(
        yTest, yTPred, average="macro", zero_division=0)
    testMetrics = {"accuracy": tAcc, "precision": tp, "recall": tr, "f1": tf,
                   "train_time": trainTime, "inference_time": infTime}
    testCm = confusion_matrix(yTest, yTPred)

    scores      = finalModel.decision_function(XTest)
    fpr, tpr, _ = roc_curve(yTest, scores)
    aucVal      = auc(fpr, tpr)

    return (tuningDf, bestC, valMetrics, valCm,
            testMetrics, testCm, (fpr, tpr), aucVal, finalModel)


def runDataset(datasetNum):
    """Run the PCA-reduced SVM pipeline on one dataset. Returns results dict."""
    trainPath = DATASET_DIR / f"dataset_{datasetNum}_train.json"

    df     = loadAndFlatten(trainPath)
    texts  = df.apply(buildText, axis=1)
    le     = LabelEncoder()
    labels = le.fit_transform(df["label"])
    labelNames = [str(c) for c in le.classes_]

    XTrTxt, XTmpTxt, yTr, yTmp = train_test_split(
        texts, labels,
        test_size=(VAL_FRAC + (1 - TRAIN_FRAC - VAL_FRAC)),
        random_state=RANDOM_STATE, stratify=labels,
    )
    XVTxt, XTeTxt, yVal, yTest = train_test_split(
        XTmpTxt, yTmp,
        test_size=0.5, random_state=RANDOM_STATE, stratify=yTmp,
    )

    tfidf = TfidfVectorizer(analyzer="char_wb",
                            ngram_range=TFIDF_NGRAM_RANGE,
                            max_features=TFIDF_MAX_FEATURES,
                            sublinear_tf=True, min_df=2)
    XTrRaw = tfidf.fit_transform(XTrTxt)
    XVRaw  = tfidf.transform(XVTxt)
    XTeRaw = tfidf.transform(XTeTxt)

    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    svd.fit(XTrRaw)

    XTrPca = normalize(svd.transform(XTrRaw))
    XVPca  = normalize(svd.transform(XVRaw))
    XTePca = normalize(svd.transform(XTeRaw))

    (pcaTuning, pcaBestC,
     pcaValMet, pcaValCm,
     pcaTestMet, pcaTestCm,
     pcaRoc, pcaAuc,
     pcaModel) = tuneAndEval(XTrPca, yTr, XVPca, yVal, XTePca, yTest)

    joblib.dump({
        "tfidf":         tfidf,
        "svd":           svd,
        "pca_model":     pcaModel,
        "label_encoder": le,
        "config": {"dataset_num": datasetNum,
                   "best_C":      pcaBestC,
                   "n_components": N_COMPONENTS},
        "metrics": {"validation": pcaValMet, "test": pcaTestMet},
    }, OUTPUT_DIR / f"svm_pca_dataset{datasetNum}.joblib")

    labelCounts = df["label"].value_counts()
    return {
        "dataset":     datasetNum,
        "df":          df,
        "svd":         svd,
        "label_names": labelNames,
        "n_benign":    int(labelCounts.get("Benign",    0)),
        "n_malicious": int(labelCounts.get("Malicious", 0)),
        "tuning": {"pca": pcaTuning},
        "pca": {
            "best_C":       pcaBestC,
            "val_metrics":  pcaValMet,
            "val_cm":       pcaValCm,
            "test_metrics": pcaTestMet,
            "test_cm":      pcaTestCm,
            "roc":          pcaRoc,
            "auc":          pcaAuc,
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    np.random.seed(RANDOM_STATE)
    sns.set_style("whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    availableNums = [n for n in ALL_DATASETS
                     if (DATASET_DIR / f"dataset_{n}_train.json").exists()]
    for n in set(ALL_DATASETS) - set(availableNums):
        print(f"[SKIP] Dataset {n} not found: {DATASET_DIR / f'dataset_{n}_train.json'}")

    print(f"Running {len(availableNums)} dataset(s) in parallel...")
    allResults = joblib.Parallel(n_jobs=len(availableNums))(
        joblib.delayed(runDataset)(n) for n in availableNums
    )
    allResults = [r for r in allResults if r is not None]

    if not allResults:
        print("No datasets found. Check that ./Datasets/ folder exists.")
        return

    for result in allResults:
        print(f"Dataset {result['dataset']}:")
        print(f"  PCA  — Test F1 = {result['pca']['test_metrics']['f1']:.4f}  "
              f"AUC = {result['pca']['auc']:.4f}  "
              f"Best C = {result['pca']['best_C']}")

    print("\nGenerating plots...")

    allDfs    = {r["dataset"]: r["df"]  for r in allResults}
    allSvds   = {r["dataset"]: r["svd"] for r in allResults}
    allTuning = {r["dataset"]: r["tuning"] for r in allResults}
    allValCms = {r["dataset"]: r["pca"]["val_cm"]  for r in allResults}
    allTestCms= {r["dataset"]: r["pca"]["test_cm"] for r in allResults}
    labelNames = allResults[0]["label_names"]

    plotAttackDistribution(allDfs)
    plotExplainedVariance(allSvds)
    plotCTuning(allTuning)
    plotConfusionMatrices(allValCms,  labelNames, "Validation",
                          OUTPUT_DIR / "cm_validation.png")
    plotConfusionMatrices(allTestCms, labelNames, "Test",
                          OUTPUT_DIR / "cm_test.png")
    plotROCCurves(allResults)
    plotTiming(allResults)

    if len(allResults) >= 2:
        plotMetrics(allResults)

    print("\nResults Summary:")
    hdr = f"  {'Dataset':<12} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}  {'Train(s)':>9}  {'Inf(ms)':>8}  Best C"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")
    for r in allResults:
        m    = r["pca"]["test_metrics"]
        aucV = r["pca"]["auc"]
        print(f"  Dataset {r['dataset']:<4}  "
              f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {aucV:>7.4f}  "
              f"{m['train_time']:>9.3f}  {m['inference_time']*1000:>8.2f}  "
              f"{r['pca']['best_C']}")

    if len(allResults) >= 2:
        f1s  = [r["pca"]["test_metrics"]["f1"] for r in allResults]
        aucs = [r["pca"]["auc"]                for r in allResults]
        print(f"\n  Avg Test F1 = {np.mean(f1s):.4f}  "
              f"(std {np.std(f1s):.4f})  |  Avg AUC = {np.mean(aucs):.4f}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


# =============================================================================
# INFERENCE UTILITY
# =============================================================================

def predict(requests, artifactPath):
    """
    Classify new HTTP request strings using a saved pipeline.

    Parameters
    ----------
    requests    : list of str  — raw joined request strings
    artifactPath: str or Path  — path to .joblib artifact saved by main()

    Returns
    -------
    list of str — "Benign" or "Malicious" for each request
    """
    art   = joblib.load(artifactPath)
    tfidf = art["tfidf"]
    le    = art["label_encoder"]
    X     = normalize(art["svd"].transform(tfidf.transform(pd.Series(requests))))
    enc   = art["pca_model"].predict(X)
    return le.inverse_transform(enc).tolist()


if __name__ == "__main__":
    main()