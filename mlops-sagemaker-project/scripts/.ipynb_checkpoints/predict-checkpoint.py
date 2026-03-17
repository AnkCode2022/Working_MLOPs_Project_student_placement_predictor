import boto3

runtime = boto3.client("sagemaker-runtime")

payload = "7.74,96,72"

response = runtime.invoke_endpoint(
    EndpointName="placement-endpoint",
    ContentType="text/csv",
    Body=payload
)

prob = float(response["Body"].read().decode())

print("Probability:", prob)

prediction = 1 if prob > 0.5 else 0

print("Final Prediction:", prediction)