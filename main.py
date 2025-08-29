"""
Capstone Project: Breast Cancer Mass Classification
---------------------------------------------------
Run either:
    python main.py --mlp      # Train & evaluate Neural Network (MLP)
    python main.py --compare  # Compare ML models (LogReg, DecisionTree, RF, SVM)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def load_mammographic_dataset(path: str):
    cols = ["birads", "age", "shape", "margin", "density", "severity"]
    df = pd.read_csv(path, header=None, names=cols)
    df = df.replace("?", np.nan)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def run_mlp(data_path: Path, out_dir: Path):
    print("Running Deep Learning Project (MLP)...")

    df = load_mammographic_dataset(data_path)
    X = df.drop(columns=["severity"])
    y = df["severity"].astype(int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(16, 8),
                              activation="relu", solver="adam",
                              max_iter=500, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "deep_learning_output.txt", "w") as f:
        f.write("Deep Learning Project (MLP) Results\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Test ROC-AUC: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, digits=4) + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n")

    RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    plt.title("MLP ROC Curve")
    plt.savefig(out_dir / "deep_learning_roc.png", bbox_inches="tight")
    plt.close()

    print("Done. Results saved in outputs/")


def run_compare(data_path: Path, out_dir: Path):
    print("Running Model Comparison Project...")

    df = load_mammographic_dataset(data_path)
    X = df.drop(columns=["severity"])
    y = df["severity"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    models = {
        "LogReg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200))
        ]),
        "DecisionTree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
        ]),
        "SVM-RBF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
    }

    results = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        if hasattr(pipe[-1], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        auc = roc_auc_score(y_test, y_proba)
        results.append({
            "name": name,
            "auc": auc,
            "report": classification_report(y_test, y_pred, digits=4),
            "cm": confusion_matrix(y_test, y_pred),
            "pipe": pipe
        })

    results.sort(key=lambda r: r["auc"], reverse=True)
    best = results[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final_project_output.txt", "w") as f:
        f.write("Final Project Assignment: Model Comparison Results\n")
        f.write("=" * 45 + "\n\n")
        for r in results:
            f.write(f"Model: {r['name']} | ROC-AUC: {r['auc']:.4f}\n")
        f.write("\nBest model: " + best["name"] + f" (ROC-AUC={best['auc']:.4f})\n\n")
        f.write("Classification Report (best):\n")
        f.write(best["report"] + "\n")
        f.write("Confusion Matrix (best):\n")
        f.write(str(best["cm"]) + "\n")

    plt.figure()
    for r in results:
        RocCurveDisplay.from_estimator(r["pipe"], X_test, y_test, name=r["name"])
    plt.title("ROC Curves - Model Comparison")
    plt.savefig(out_dir / "final_project_roc.png", bbox_inches="tight")
    plt.close()

    print("Done. Results saved in outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breast Cancer Mass Classification Project")
    parser.add_argument("--mlp", action="store_true", help="Run Deep Learning Project (MLP)")
    parser.add_argument("--compare", action="store_true", help="Run Model Comparison Project")
    parser.add_argument("--data", type=str, default="data/mammographic_masses.data.txt",
                        help="Path to dataset file")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)

    if args.mlp:
        run_mlp(data_path, out_dir)
    elif args.compare:
        run_compare(data_path, out_dir)
    else:
        print("Please specify --mlp or --compare")
