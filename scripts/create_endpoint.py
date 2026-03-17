import boto3

client = boto3.client("sagemaker")

client.create_endpoint(
    EndpointName="placement-endpoint",
    EndpointConfigName="placement-endpoint-config"
)

print("Endpoint deployment started")