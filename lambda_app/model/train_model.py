"""Train a logistic regression model for breast cancer classification.

This script demonstrates the offline training workflow. It downloads the
Breast Cancer Wisconsin dataset from ``scikit-learn`` and trains a
regularised logistic regression model. The model coefficients are stored
as JSON so that the AWS Lambda runtime can load them without depending on
``scikit-learn`` or ``numpy``.

Usage
-----
python -m lambda_app.model.train_model
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .predict import ModelParameters


@dataclass
class TrainingMetadata:
    accuracy: float
    roc_auc: float
    trained_at: str
    feature_names: List[str]
    target_names: List[str]

    def to_dict(self) -> Dict[str, str]:
        data = asdict(self)
        data["accuracy"] = f"{self.accuracy:.4f}"
        data["roc_auc"] = f"{self.roc_auc:.4f}"
        return data


def train() -> ModelParameters:
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42, stratify=dataset.target
    )

    model = LogisticRegression(max_iter=500, solver="liblinear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metadata = TrainingMetadata(
        accuracy=accuracy_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_prob),
        trained_at=datetime.now(timezone.utc).isoformat(),
        feature_names=list(dataset.feature_names),
        target_names=[str(name) for name in dataset.target_names],
    )

    params = ModelParameters(
        intercept=model.intercept_.tolist(),
        coefficients=model.coef_.tolist(),
        classes=model.classes_.tolist(),
        feature_names=list(dataset.feature_names),
        metadata={
            "accuracy": metadata.to_dict()["accuracy"],
            "roc_auc": metadata.to_dict()["roc_auc"],
            "trained_at": metadata.trained_at,
            "target_names": ", ".join(metadata.target_names),
        },
    )
    return params


def save_parameters(parameters: ModelParameters, output_path: Path) -> None:
    payload = {
        "intercept": parameters.intercept,
        "coefficients": parameters.coefficients,
        "classes": parameters.classes,
        "feature_names": parameters.feature_names,
        "metadata": parameters.metadata,
    }
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def main() -> None:
    parameters = train()
    output_path = Path(__file__).resolve().parent / "model_parameters.json"
    save_parameters(parameters, output_path)
    print(f"Saved parameters to {output_path}")


if __name__ == "__main__":
    main()
