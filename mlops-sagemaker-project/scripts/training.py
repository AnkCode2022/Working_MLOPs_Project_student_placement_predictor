# import pandas as pd
# import xgboost as xgb
# import os

# df = pd.read_csv("data/train.csv")

# X = df.drop("placed", axis=1)
# y = df["placed"]

# model = xgb.XGBClassifier(
#     n_estimators=100,
#     max_depth=3,
#     learning_rate=0.1
# )

# model.fit(X, y)

# os.makedirs("model", exist_ok=True)

# # Save as booster (required by SageMaker container)
# booster = model.get_booster()
# booster.save_model("model/xgboost-model")

# print("Model trained and saved")
# scripts/training.py
# scripts/training.py
import pandas as pd
import xgboost as xgb
import joblib
import os

train_path = "/opt/ml/input/data/train/train.csv"

print("Reading training data:", train_path)

df = pd.read_csv(train_path)

y = df.iloc[:,0]
X = df.iloc[:,1:]

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss"
)

model.fit(X,y)

model_dir = "/opt/ml/model"

os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir,"model.pkl"))

print("Training completed")