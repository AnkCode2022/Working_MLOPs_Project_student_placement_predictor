import pickle
import json
import os

def model_fn(model_dir):

    with open(os.path.join(model_dir,"model.pkl"),"rb") as f:
        model = pickle.load(f)

    return model

def input_fn(request_body, content_type):

    data = json.loads(request_body)

    return data

def predict_fn(input_data, model):

    prediction = model.predict([[
        input_data["cgpa"],
        input_data["iq"],
        input_data["profile_score"]
    ]])

    return prediction.tolist()

def output_fn(prediction, content_type):

    return json.dumps(prediction)