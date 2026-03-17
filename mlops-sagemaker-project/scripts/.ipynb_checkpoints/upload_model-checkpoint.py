import boto3

client = boto3.client("sagemaker")

MODEL_PACKAGE_GROUP = "placement-model-group"

response = client.create_model_package(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelPackageDescription="Placement prediction model",

    InferenceSpecification={
        "Containers": [
            {
                "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1",
                "ModelDataUrl": "s3://placement-project-bkt/model/model.tar.gz"
            }
        ],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"]
    }
)

print("Model registered")