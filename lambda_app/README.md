# AWS Lambda Breast Cancer Classifier

This project demonstrates how to train and deploy a lightweight machine
learning model on AWS Lambda. A logistic regression model is trained on
the open Breast Cancer Wisconsin dataset. The resulting coefficients are
stored in JSON so that the Lambda function can serve predictions without
requiring heavy numerical dependencies.

## Repository structure

```
lambda_app/
├── lambda_function.py   # AWS Lambda handler
├── model/
│   ├── model_parameters.json  # Pre-trained weights stored as JSON
│   ├── predict.py        # Lightweight inference implementation
│   └── train_model.py    # Offline training script
├── tests/
│   └── test_lambda.py    # Basic unit tests
└── README.md             # This file
```

## Prerequisites

* Python 3.11+
* AWS SAM CLI (for local testing and deployment)
* AWS account with permissions to deploy Lambda and API Gateway

## 1. Train the model

Training is performed locally. The script downloads the dataset from
`scikit-learn`, trains a logistic regression model, and saves the
parameters to `model/model_parameters.json`.

```bash
python -m pip install -r requirements-training.txt
python -m lambda_app.model.train_model
```

The repository already contains pre-generated parameters, so this step is
only necessary when you want to retrain the model.

## 2. Run unit tests

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## 3. Package for AWS Lambda

1. Create a deployment package (no external runtime dependencies are
   required because inference uses only the Python standard library).

```bash
zip -r9 lambda_package.zip lambda_app
```

2. Alternatively, use AWS SAM to build and deploy:

```bash
sam build --template deployment/template.yaml
sam deploy --guided
```

SAM handles uploading the application code to S3, creating the Lambda
function, and provisioning an API Gateway endpoint.

## 4. Invoke the API

Once deployed, you can send HTTP POST requests with the feature vector.
You can supply the features as either an ordered list or a JSON object
with named features. The example below uses the ordered list approach.

```bash
curl -X POST "https://<api-id>.execute-api.<region>.amazonaws.com/Prod/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
      0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
      15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
    ]
  }'
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.22,
  "model_metadata": {
    "accuracy": "0.96",
    "roc_auc": "0.99",
    "trained_at": "2024-01-01T00:00:00+00:00",
    "target_names": "malignant, benign"
  }
}
```

## 5. Clean up

When you are done testing, remove the deployed resources to avoid
incurring costs:

```bash
sam delete --stack-name <stack-name>
```

## Notes on production readiness

* The Lambda handler performs basic input validation and returns helpful
  error messages for missing data.
* The model parameters are compact (~10 KB) and can be loaded directly
  from the Lambda deployment package.
* For production workloads, consider storing the parameters in Amazon S3
  and enabling encrypted environment variables for configuration.
* Add monitoring using Amazon CloudWatch metrics and configure alarms for
  error rates or latency.
