from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai_resume_screening.csv"
MODEL_PATH = BASE_DIR / "resume_screening_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

TARGET = "shortlisted"

NUMERIC_FEATURES = [
    "years_experience",
    "skills_match_score",
    "project_count",
    "resume_length",
    "github_activity",
]

CATEGORICAL_FEATURES = ["education_level"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline():
    # Scaling is not required for Random Forest, so keep numeric values unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required_columns = set(FEATURES + [TARGET])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    target_map = {"Yes": 1, "No": 0}
    y = df[TARGET].map(target_map)

    if y.isna().any():
        invalid = sorted(df.loc[y.isna(), TARGET].astype(str).unique())
        raise ValueError(f"Unexpected target values: {invalid}")

    X = df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not Shortlisted", "Shortlisted"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": "RandomForestClassifier",
        "test_size": 0.20,
        "random_state": 42,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "features": FEATURES,
    }

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
