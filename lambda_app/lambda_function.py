"""AWS Lambda handler for a lightweight binary classifier.

The function expects an API Gateway proxy event whose JSON body contains
an object with a ``features`` key. The value associated with ``features``
can be either a list of feature values or a dictionary that maps feature
names to values. The features are mapped to the order that the model was
trained with. The handler returns the predicted class along with the
associated probability for the positive class.
"""
from __future__ import annotations

import json
from http import HTTPStatus
from typing import Dict

from .model.predict import BreastCancerRiskModel, ModelLoadingError


MODEL: BreastCancerRiskModel | None = None


def _get_model() -> BreastCancerRiskModel:
    global MODEL
    if MODEL is None:
        MODEL = BreastCancerRiskModel.from_disk()
    return MODEL


def lambda_handler(event: Dict, _context) -> Dict:
    """Entrypoint for AWS Lambda.

    Parameters
    ----------
    event:
        The AWS Lambda event. When invoked through API Gateway this is the
        proxy event which includes the request body.
    _context:
        Lambda context object (unused).

    Returns
    -------
    dict
        API Gateway compatible response containing the prediction.
    """

    try:
        body = event.get("body") if isinstance(event, dict) else None
        if body is None:
            raise ValueError("Missing request body")

        if isinstance(body, str):
            body = json.loads(body or "{}")

        features = body.get("features")
        if features is None:
            raise ValueError("Missing 'features' in request body")

        model = _get_model()
        ordered_features = model.prepare_features(features)
        prediction, probability = model.predict(ordered_features)

        response = {
            "prediction": prediction,
            "probability": probability,
            "model_metadata": model.metadata,
        }

        return {
            "statusCode": HTTPStatus.OK,
            "body": json.dumps(response),
            "headers": {"Content-Type": "application/json"},
        }

    except (ValueError, KeyError) as exc:
        return {
            "statusCode": HTTPStatus.BAD_REQUEST,
            "body": json.dumps({"error": str(exc)}),
            "headers": {"Content-Type": "application/json"},
        }
    except ModelLoadingError as exc:
        return {
            "statusCode": HTTPStatus.INTERNAL_SERVER_ERROR,
            "body": json.dumps({"error": f"Model loading error: {exc}"}),
            "headers": {"Content-Type": "application/json"},
        }
