# import boto3
# import pandas as pd
# import os

# BUCKET = "placement-project-bkt"
# KEY = "rowdata/students_placement.csv"

# os.makedirs("data", exist_ok=True)

# s3 = boto3.client("s3")

# s3.download_file(
#     BUCKET,
#     KEY,
#     "data/raw.csv"
# )

# print("Dataset downloaded from S3")


import pandas as pd
import os

input_dir = "/opt/ml/processing/input"
output_dir = "/opt/ml/processing/output"

os.makedirs(output_dir, exist_ok=True)

file_name = os.listdir(input_dir)[0]
input_path = os.path.join(input_dir, file_name)

print("Reading dataset:", input_path)

df = pd.read_csv(input_path)

print("Dataset shape:", df.shape)

df.to_csv(os.path.join(output_dir, "raw.csv"), index=False)

print("Data ingestion completed")