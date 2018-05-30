from keras.models import Sequential, model_from_json

import numpy, sys

def load_model(file):
    json_file = open(file, "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)

    return loaded_model
    
def save_model(file, model):
    model_json = model.to_json()
    with open(file, "w") as json_file:
        json_file.write(model_json)
    
