import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

test_path = "/opt/ml/processing/test/test.csv"
model_path = "/opt/ml/processing/model/model.pkl"

df = pd.read_csv(test_path)

y = df.iloc[:,0]
X = df.iloc[:,1:]

model = joblib.load(model_path)

pred = model.predict(X)

acc = accuracy_score(y,pred)

print("Model Accuracy:",acc)