# import pandas as pd
# import joblib
# import json
# import tarfile
# import os
# from sklearn.metrics import accuracy_score

# # Extract model artifact
# model_tar = "/opt/ml/processing/model/model.tar.gz"

# with tarfile.open(model_tar) as tar:
#     tar.extractall("/opt/ml/processing/model")

# # Load extracted model
# model_path = "/opt/ml/processing/model/model.pkl"
# model = joblib.load(model_path)

# # Load test data
# test_path = "/opt/ml/processing/test/test.csv"
# df = pd.read_csv(test_path)

# X = df[["cgpa","iq","profile_score"]]
# y = df["placed"]

# # Predict
# preds = model.predict(X)

# acc = accuracy_score(y, preds)

# # metrics = {"accuracy": acc}
# metrics = {
#     "classification_metrics": {
#         "accuracy": {
#             "value": acc
#         }
#     }
# }

# # Save evaluation output
# os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)

# with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
#     json.dump(metrics, f)

# print(metrics)

import pandas as pd
import json
import tarfile
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score

# -------------------------
# Extract model artifact
# -------------------------
model_tar = "/opt/ml/processing/model/model.tar.gz"

with tarfile.open(model_tar) as tar:
    tar.extractall("/opt/ml/processing/model")

# -------------------------
# Load extracted XGBoost model
# -------------------------
model_path = "/opt/ml/processing/model/xgboost-model"

model = xgb.Booster()
model.load_model(model_path)

# -------------------------
# Load test data
# -------------------------
test_path = "/opt/ml/processing/test/test.csv"
df = pd.read_csv(test_path)

X = df[["cgpa", "iq", "profile_score"]]
y = df["placed"]

# -------------------------
# Convert to DMatrix (required for Booster)
# -------------------------
dtest = xgb.DMatrix(X)

# -------------------------
# Predict
# -------------------------
preds = model.predict(dtest)

# Convert probabilities → class labels
preds = [1 if p > 0.5 else 0 for p in preds]

# -------------------------
# Evaluate
# -------------------------
acc = accuracy_score(y, preds)

metrics = {
    "classification_metrics": {
        "accuracy": {
            "value": acc
        }
    }
}

# -------------------------
# Save evaluation output
# -------------------------
os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)

with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    json.dump(metrics, f)

print(metrics)