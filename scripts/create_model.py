import boto3
import sagemaker
from sagemaker.image_uris import retrieve

session = sagemaker.Session()
region = session.boto_region_name

image_uri = retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1"
)

client = boto3.client("sagemaker")

response = client.create_model(
    ModelName="placement-xgb-model",

    PrimaryContainer={
        "Image": image_uri,
        "ModelDataUrl": "s3://placement-project-bkt/model/model.tar.gz"
    },

    ExecutionRoleArn=sagemaker.get_execution_role()
)

print("Model created successfully")