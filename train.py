import argparse
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def make_demo_data(n=5000, random_state=42):
    """Synthetic data to verify the project runs; not real fraud data."""
    X, y = make_classification(
        n_samples=n, n_features=10, n_informative=6, n_redundant=2,
        n_clusters_per_class=2, weights=[0.985, 0.015],
        class_sep=1.2, random_state=random_state
    )
    columns = [f"Feature_{i}" for i in range(1, 11)]
    df = pd.DataFrame(X, columns=columns)
    df["Amount"] = np.random.default_rng(random_state).exponential(60, n)
    df["Class"] = y
    return df

def load_data(path):
    if path and Path(path).exists():
        print(f"Loading dataset: {path}")
        return pd.read_csv(path), True
    print("No dataset file found. Creating synthetic demonstration data.")
    print("For meaningful results, use the Kaggle dataset (see README.md).")
    return make_demo_data(), False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv",
                        help="Path to CSV with target column Class")
    args = parser.parse_args()

    df, is_real = load_data(args.data)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a target column named 'Class'.")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Class"]).copy()
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    # Remove non-numeric columns and fill missing numeric values.
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.fillna(X.median(numeric_only=True))
    if y.nunique() < 2:
        raise ValueError("The target column must contain both classes 0 and 1.")

    print("\nDataset shape:", df.shape)
    print("\nClass counts:\n", y.value_counts().sort_index())
    print("\nFraud percentage: {:.4f}%".format(100 * y.mean()))

    # EDA charts
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Transaction Class Distribution (0 = Legitimate, 1 = Fraud)")
    plt.xlabel("Class")
    plt.ylabel("Number of transactions")
    plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=150)
    plt.close()

    if "Amount" in X.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x="Amount", hue="Class", bins=50,
                     element="step", stat="density", common_norm=False)
        plt.title("Transaction Amount Distribution")
        plt.tight_layout()
        plt.savefig(OUT / "amount_distribution.png", dpi=150)
        plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1, min_samples_leaf=2
        )
    }

    rows = []
    best_name, best_model, best_score = None, None, -1
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, score)
        pr = average_precision_score(y_test, score)
        report = classification_report(y_test, pred, output_dict=True,
                                       zero_division=0)
        rows.append({
            "Model": name,
            "Accuracy": report["accuracy"],
            "Precision (fraud)": report["1"]["precision"],
            "Recall (fraud)": report["1"]["recall"],
            "F1-score (fraud)": report["1"]["f1-score"],
            "ROC-AUC": roc,
            "PR-AUC": pr
        })
        print(classification_report(y_test, pred, zero_division=0))
        print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

        # Select using PR-AUC, more informative for imbalanced fraud data.
        if pr > best_score:
            best_name, best_model, best_score = name, model, pr

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Legitimate", "Fraud"],
            cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(f"{name} - Confusion Matrix")
        fig.tight_layout()
        fig.savefig(OUT / (name.lower().replace(" ", "_") + "_confusion_matrix.png"),
                    dpi=150)
        plt.close(fig)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON\n", results.to_string(index=False))

    # Comparison plot
    plot_df = results.set_index("Model")[["Precision (fraud)", "Recall (fraud)", "F1-score (fraud)"]]
    ax = plot_df.plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    ax.set_title("Fraud-class model comparison")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close()

    joblib.dump({
        "model": best_model,
        "features": list(X.columns),
        "model_name": best_name,
        "is_real_dataset": is_real
    }, OUT / "fraud_model.joblib")

    metadata = {
        "model_name": best_name,
        "features": list(X.columns),
        "is_real_dataset": is_real,
        "fraud_rate": float(y.mean())
    }
    pd.Series(metadata, dtype=object).to_json(OUT / "model_metadata.json", indent=2)
    print(f"\nSaved best model: {best_name}")
    print("Outputs saved in:", OUT)
    if not is_real:
        print("\nNOTE: These results are from synthetic demo data, not real transactions.")

if __name__ == "__main__":
    main()
