import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--input-data")
parser.add_argument("--train-output")
parser.add_argument("--test-output")

args = parser.parse_args()

df = pd.read_csv(args.input_data)

X = df[["cgpa","iq","profile_score"]]
y = df["placed"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

train_df = X_train.copy()
train_df["placed"] = y_train

test_df = X_test.copy()
test_df["placed"] = y_test

train_df.to_csv(args.train_output,index=False)
test_df.to_csv(args.test_output,index=False)

print("Preprocessing finished")