from keras.models import Sequential, model_from_json

from keras.layers import Dense, Activation, Flatten, Convolution2D, Convolution3D, AveragePooling2D, Reshape, Lambda, Permute

from keras.utils import np_utils

from keras import backend as K

import numpy as np

import sys, json

from grnnf import model_manip, data_manip

data_dict = data_manip.load_data("../preprocessed_data/ppd.json")

X_train = np.array(data_dict["X_train"])
Y_train = np.array(data_dict["Y_train"])
X_test = np.array(data_dict["X_test"])
Y_test = np.array(data_dict["Y_test"])

#load model structure from json
model_file = sys.argv[1]

loaded_model = model_manip.load_model(model_file)

#load weights into loaded model
weight_file = sys.argv[2]
loaded_model.load_weights(weight_file)

loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
train_score = loaded_model.evaluate(X_train, Y_train, verbose=0)
test_score = loaded_model.evaluate(X_test, Y_test, verbose=0)

print(train_score)
print(test_score)
