# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os

# df = pd.read_csv("data/raw.csv")

# X = df.drop("placed", axis=1)
# y = df["placed"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )

# train = X_train.copy()
# train["placed"] = y_train

# test = X_test.copy()
# test["placed"] = y_test

# os.makedirs("data", exist_ok=True)

# train.to_csv("data/train.csv", index=False)
# test.to_csv("data/test.csv", index=False)

# print("Train and test saved")

# scripts/preprocessing.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

input_dir = "/opt/ml/processing/input"
file_name = os.listdir(input_dir)[0]

df = pd.read_csv(os.path.join(input_dir, file_name))

print("Columns:", df.columns)

X = df.drop(columns=["placed"])
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_dir = "/opt/ml/processing/train"
test_dir = "/opt/ml/processing/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

pd.concat([y_train, X_train], axis=1).to_csv(
    os.path.join(train_dir, "train.csv"), index=False
)

pd.concat([y_test, X_test], axis=1).to_csv(
    os.path.join(test_dir, "test.csv"), index=False
)

print("Preprocessing completed")