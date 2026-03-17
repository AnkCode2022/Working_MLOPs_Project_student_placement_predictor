import joblib
import os
import pandas as pd
from io import StringIO

# Load model
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model

# Parse input
def input_fn(request_body, content_type):
    if content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body))
        return df
    else:
        raise ValueError("Unsupported content type")

# Predict
def predict_fn(input_data, model):
    return model.predict(input_data)

# Format output
def output_fn(prediction, accept):
    return str(prediction)