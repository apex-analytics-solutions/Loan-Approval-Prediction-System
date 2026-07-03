"""
Loan Prediction — Training Pipeline.

Trains LR / Decision Tree / Random Forest / XGBoost, evaluates each,
saves the best model artifact and all comparison outputs.

Usage:
    python train.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)

from utils import (
    setup_logging, ensure_dirs,
    DATASET_PATH, ARTIFACTS_PATH, MODEL_COMPARISON_CSV,
    FEATURE_IMPORTANCE_CSV, DASHBOARD_DATA_PATH, OUTPUTS_DIR,
    TARGET, RANDOM_STATE, TEST_SIZE, VERSION,
    save_artifacts,
)

warnings.filterwarnings("ignore")

log = setup_logging("train")


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "Accuracy":  round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall":    round(recall_score(y_test, pred), 4),
        "F1 Score":  round(f1_score(y_test, pred), 4),
        "ROC AUC":   round(roc_auc_score(y_test, proba), 4) if proba is not None else 0.0,
    }


def plot_bar(data, column, colors, filename, title):
    plt.figure(figsize=(9, 5))
    bars = plt.bar(data.index, data[column], color=colors)
    plt.title(title, fontsize=16, weight="bold")
    plt.ylabel(column)
    plt.ylim(0, 1)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                 ha="center", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    log.info("=" * 60)
    log.info("LOAN PREDICTION TRAINING PIPELINE  v%s", VERSION)
    log.info("=" * 60)

    # ── Load Dataset ──
    df = pd.read_csv(str(DATASET_PATH))
    log.info("Dataset loaded: %s", df.shape)

    # ── Missing Values ──
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    log.info("Missing values handled")

    # ── Encoding ──
    encoders = {}
    categorical = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    log.info("Categorical encoding done: %d columns", len(encoders))

    # ── Feature Engineering ──
    if "monthly_income" in df.columns:
        df["monthly_income_log"] = np.log1p(df["monthly_income"])
    if "loan_amount" in df.columns:
        df["loan_amount_log"] = np.log1p(df["loan_amount"])

    drop_cols = [TARGET]
    if "monthly_income" in df.columns:
        drop_cols.append("monthly_income")
    if "loan_amount" in df.columns:
        drop_cols.append("loan_amount")

    X = df.drop(drop_cols, axis=1)
    y = df[TARGET]
    feature_names = X.columns.tolist()
    log.info("Features: %d  |  Samples: %d", len(feature_names), len(X))

    # ── Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    log.info("Train: %s  |  Test: %s", X_train.shape, X_test.shape)

    # ── Models ──
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )
        log.info("XGBoost detected — included")
    except ImportError:
        log.warning("XGBoost not installed — skipping")

    # ── Train & Evaluate ──
    results = []
    trained_models = {}

    for name, model_obj in models.items():
        model_obj.fit(X_train, y_train)
        scores = evaluate_model(model_obj, X_test, y_test)
        scores["Model"] = name
        results.append(scores)
        trained_models[name] = model_obj

        log.info(
            "%-22s  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  AUC=%.4f",
            name, scores["Accuracy"], scores["Precision"],
            scores["Recall"], scores["F1 Score"], scores["ROC AUC"],
        )

    # ── Select Best Model ──
    comparison = pd.DataFrame(results)
    col_order = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    comparison = comparison[col_order].sort_values(
        by=["ROC AUC", "Accuracy"], ascending=[False, False],
    ).reset_index(drop=True)

    best_row = comparison.iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]

    log.info("=" * 60)
    log.info("BEST MODEL: %s  (AUC=%.4f, Acc=%.4f)",
             best_name, best_row["ROC AUC"], best_row["Accuracy"])
    log.info("=" * 60)

    # ── Save Comparison CSV ──
    comparison.to_csv(str(MODEL_COMPARISON_CSV), index=False)
    log.info("Saved %s", MODEL_COMPARISON_CSV.name)

    # ── Save Model Artifact ──
    artifacts = {
        "model": best_model,
        "feature_names": feature_names,
        "encoders": encoders,
        "best_model_name": best_name,
    }
    save_artifacts(artifacts, ARTIFACTS_PATH, log)

    # ── Confusion Matrix ──
    best_predictions = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_predictions)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Confusion Matrix", fontsize=16, weight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "confusion_matrix.png"), dpi=300)
    plt.close()
    log.info("Saved confusion_matrix.png")

    # ── Classification Report ──
    report = classification_report(y_test, best_predictions)
    with open(str(OUTPUTS_DIR / "classification_report.txt"), "w") as f:
        f.write(report)

    # ── ROC Curve (Best Model) ──
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_score = roc_auc_score(y_test, probabilities)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=3, label=f"AUC = {auc_score:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve", fontsize=16, weight="bold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(OUTPUTS_DIR / "roc_curve.png"), dpi=300)
        plt.close()
        log.info("Saved roc_curve.png")

    # ── Feature Importance ──
    if hasattr(best_model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": best_model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)

        importance.to_csv(str(FEATURE_IMPORTANCE_CSV), index=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance.head(15), x="Importance", y="Feature", palette="viridis")
        plt.title("Top 15 Important Features", fontsize=16, weight="bold")
        plt.tight_layout()
        plt.savefig(str(OUTPUTS_DIR / "feature_importance.png"), dpi=300)
        plt.close()
        log.info("Saved feature_importance.csv + .png")

    # ── Dashboard Data (ROC curves + confusion matrices for all models) ──
    dashboard_data = {
        "roc_curves": {},
        "confusion_matrices": {},
        "feature_importance": {},
        "best_model_name": best_name,
    }

    for name, m in trained_models.items():
        preds = m.predict(X_test)
        dashboard_data["confusion_matrices"][name] = confusion_matrix(y_test, preds).tolist()

        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_test)[:, 1]
            fpr_m, tpr_m, _ = roc_curve(y_test, proba)
            auc_m = roc_auc_score(y_test, proba)
            dashboard_data["roc_curves"][name] = {
                "fpr": fpr_m.tolist(),
                "tpr": tpr_m.tolist(),
                "auc": round(auc_m, 4),
            }

    if hasattr(best_model, "feature_importances_"):
        dashboard_data["feature_importance"] = dict(
            zip(feature_names, best_model.feature_importances_.tolist())
        )

    with open(str(DASHBOARD_DATA_PATH), "wb") as f:
        pickle.dump(dashboard_data, f)
    log.info("Saved dashboard_data.pkl (ROC curves + confusion matrices for %d models)",
             len(trained_models))

    # ── Performance Charts ──
    sns.set_style("whitegrid")
    metrics = comparison.set_index("Model")
    n_models = len(metrics)

    for metric, palette_name in [
        ("Accuracy", "Set2"), ("Precision", "Set1"),
        ("Recall", "Dark2"), ("F1 Score", "Pastel1"),
        ("ROC AUC", "coolwarm"),
    ]:
        plot_bar(
            metrics, metric,
            sns.color_palette(palette_name, n_models),
            str(OUTPUTS_DIR / f"{metric.lower().replace(' ', '_')}_comparison.png"),
            f"{metric} Comparison",
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    metrics.plot(kind="bar", ax=ax)
    plt.title("Overall Model Performance", fontsize=17, weight="bold")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "model_dashboard.png"), dpi=300)
    plt.close()
    log.info("Performance charts saved")

    # ── Leaderboard ──
    leaderboard = comparison.copy().sort_values(
        by=["ROC AUC", "Accuracy"], ascending=[False, False],
    ).reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1
    leaderboard.index.name = "Rank"
    leaderboard.to_csv(str(OUTPUTS_DIR / "model_leaderboard.csv"))

    # ── Cross Validation ──
    log.info("Running 5-fold cross-validation on %s ...", best_name)
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
    pd.DataFrame([{
        "CV Mean Accuracy": cv_scores.mean(),
        "CV Std": cv_scores.std(),
    }]).to_csv(str(OUTPUTS_DIR / "cv_report.csv"), index=False)
    log.info("CV Accuracy: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())

    # ── Precision-Recall Curve ──
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X_test)[:, 1]
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color="purple", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve", fontsize=16, weight="bold")
        plt.grid()
        plt.savefig(str(OUTPUTS_DIR / "precision_recall_curve.png"), dpi=300)
        plt.close()

    # ── Scorecard ──
    scorecard = comparison.copy()
    scorecard["Overall Score"] = (
        scorecard["ROC AUC"] * 0.30 + scorecard["Accuracy"] * 0.25
        + scorecard["Precision"] * 0.15 + scorecard["Recall"] * 0.15
        + scorecard["F1 Score"] * 0.15
    )
    scorecard.sort_values(by="Overall Score", ascending=False).to_csv(
        str(OUTPUTS_DIR / "model_scorecard.csv"), index=False,
    )

    # ── Best Model Info ──
    pd.DataFrame([{
        "Best Model": best_name,
        "ROC AUC": best_row["ROC AUC"],
        "Accuracy": best_row["Accuracy"],
        "Features": len(feature_names),
        "Training Samples": len(X_train),
        "Testing Samples": len(X_test),
    }]).to_csv(str(OUTPUTS_DIR / "best_model_info.csv"), index=False)

    # ── Features Used ──
    pd.DataFrame({"Feature Name": feature_names}).to_csv(
        str(OUTPUTS_DIR / "features_used.csv"), index=False,
    )

    # ── Final Summary ──
    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("  Best Model      : %s", best_name)
    log.info("  ROC AUC         : %.4f", best_row["ROC AUC"])
    log.info("  Accuracy        : %.4f", best_row["Accuracy"])
    log.info("  Features        : %d", len(feature_names))
    log.info("  Output files    : %d", len(os.listdir(str(OUTPUTS_DIR))))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
