import boto3

client = boto3.client("sagemaker")

response = client.create_endpoint_config(
    EndpointConfigName="placement-endpoint-config",

    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": "placement-xgb-model",
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1
        }
    ]
)

print("Endpoint config created")