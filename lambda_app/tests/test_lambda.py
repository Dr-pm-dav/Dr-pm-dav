import json

from lambda_app.lambda_function import lambda_handler, _get_model


def test_model_prediction_from_list():
    model = _get_model()
    sample = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
              0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
              15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
    prediction, probability = model.predict(sample)
    assert prediction in (0, 1)
    assert 0.0 <= probability <= 1.0


def test_lambda_handler_with_dict_payload():
    model = _get_model()
    feature_names = model._params.feature_names  # pylint: disable=protected-access
    features = {name: 0.5 for name in feature_names}

    event = {"body": json.dumps({"features": features})}
    response = lambda_handler(event, None)
    body = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert "prediction" in body
    assert "probability" in body
    assert "model_metadata" in body
