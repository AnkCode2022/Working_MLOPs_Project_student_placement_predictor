import boto3

BUCKET = "placement-project-bkt"

DATA_KEY = "rowdata/students_placement.csv"

MODEL_S3_PATH = "model/model.tar.gz"

REGION = boto3.Session().region_name

MODEL_GROUP_NAME = "placement-model-group"

MODEL_NAME = "placement-xgboost-model"

ENDPOINT_CONFIG = "placement-endpoint-config"

ENDPOINT_NAME = "placement-endpoint"