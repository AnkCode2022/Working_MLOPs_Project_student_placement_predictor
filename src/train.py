# import pandas as pd
# import joblib
# from xgboost import XGBClassifier

# # training data location inside container
# train_path = "/opt/ml/input/data/train/train.csv" # this is not being used as we have path the inout file path from pipeline

# # read dataset
# df = pd.read_csv(train_path)

# X = df[["cgpa","iq","profile_score"]]
# y = df["placed"]

# # train model
# model = XGBClassifier()

# model.fit(X, y)

# # save model
# joblib.dump(model, "/opt/ml/model/model.pkl")

# print("Training completed")

import pandas as pd
from xgboost import XGBClassifier

# -------------------------
# Load training data
# -------------------------
train_path = "/opt/ml/input/data/train/train.csv"

df = pd.read_csv(train_path)

X = df[["cgpa", "iq", "profile_score"]]
y = df["placed"]

# -------------------------
# Train model
# -------------------------
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X, y)

# -------------------------
# Save in native XGBoost format
# -------------------------
model.get_booster().save_model("/opt/ml/model/xgboost-model")

print("Training completed")