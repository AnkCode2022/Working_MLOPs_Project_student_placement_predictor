import boto3
import sagemaker
from sagemaker.image_uris import retrieve

session = sagemaker.Session()
region = session.boto_region_name

# Get correct XGBoost image for your region
image_uri = retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1"
)

client = boto3.client("sagemaker")
response = client.create_model_package(
    ModelPackageGroupName="placement-model-group",
    ModelPackageDescription="Placement prediction model",
    InferenceSpecification={
        "Containers": [
            {
                "Image": image_uri,
                "ModelDataUrl": "s3://placement-project-bkt/model/model.tar.gz"
            }
        ],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"]
    }
)
print("Model registered successfully")