"""Utility module for loading the trained model parameters and generating
predictions.

The training process is handled offline (see ``train_model.py``). During
training we persist the coefficients of a logistic regression model to a
lightweight JSON file so that the Lambda function can perform inference
without heavy third-party dependencies.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

__all__ = ["BreastCancerRiskModel", "ModelLoadingError"]


class ModelLoadingError(RuntimeError):
    """Raised when the model cannot be loaded from disk."""


@dataclass(frozen=True)
class ModelParameters:
    intercept: List[float]
    coefficients: List[List[float]]
    classes: List[int]
    feature_names: List[str]
    metadata: Dict[str, str]

    @classmethod
    def from_json(cls, data: Dict) -> "ModelParameters":
        required = {"intercept", "coefficients", "classes", "feature_names"}
        missing = required - data.keys()
        if missing:
            raise ModelLoadingError(f"Missing keys in model parameters: {sorted(missing)}")
        return cls(
            intercept=list(map(float, data["intercept"])),
            coefficients=[[float(x) for x in row] for row in data["coefficients"]],
            classes=[int(cls_) for cls_ in data["classes"]],
            feature_names=list(map(str, data["feature_names"])),
            metadata=data.get("metadata", {}),
        )


class BreastCancerRiskModel:
    """Binary classifier built from pre-trained logistic regression weights."""

    PARAMETERS_FILE = Path(__file__).resolve().parent / "model_parameters.json"

    def __init__(self, parameters: ModelParameters) -> None:
        self._params = parameters

    @property
    def metadata(self) -> Dict[str, str]:
        return self._params.metadata

    @classmethod
    def from_disk(cls) -> "BreastCancerRiskModel":
        try:
            with cls.PARAMETERS_FILE.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except FileNotFoundError as exc:
            raise ModelLoadingError("Model parameters file not found") from exc
        except json.JSONDecodeError as exc:
            raise ModelLoadingError("Model parameters file is corrupted") from exc
        return cls(ModelParameters.from_json(data))

    def prepare_features(self, features: Sequence[float] | Dict[str, float]) -> List[float]:
        """Reorder/validate the features to align with training order."""
        if isinstance(features, dict):
            missing = [name for name in self._params.feature_names if name not in features]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            return [float(features[name]) for name in self._params.feature_names]

        if not isinstance(features, Iterable):
            raise ValueError("Features must be a list or mapping")

        features_list = list(features)
        if len(features_list) != len(self._params.feature_names):
            raise ValueError(
                "Incorrect number of features: "
                f"expected {len(self._params.feature_names)}, got {len(features_list)}"
            )
        return [float(value) for value in features_list]

    def _logits(self, ordered_features: Sequence[float]) -> List[float]:
        logits = []
        for intercept, coef_row in zip(self._params.intercept, self._params.coefficients):
            z = intercept
            for coef, feature in zip(coef_row, ordered_features):
                z += coef * feature
            logits.append(z)
        return logits

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-value))

    def predict(self, ordered_features: Sequence[float]) -> tuple[int, float]:
        logits = self._logits(ordered_features)
        if len(logits) == 1:
            probability = self._sigmoid(logits[0])
            prediction = self._params.classes[int(probability >= 0.5)]
            return prediction, probability

        # multi-class case (not used here but kept for completeness)
        exp_values = [math.exp(logit) for logit in logits]
        total = sum(exp_values)
        probabilities = [value / total for value in exp_values]
        max_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return self._params.classes[max_index], probabilities[max_index]
