from keras.models import Sequential, model_from_json

from keras.layers import Dense, Activation, Flatten, Convolution2D, Convolution3D, AveragePooling2D, Reshape, Lambda, Permute

from keras.utils import np_utils

from keras import backend as K

import numpy as np

import sys, json, argparse

from grnnf import model_manip, data_manip

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--description", required = True, help = "Provide a descriptor for this new neural network structure")
ap.add_argument("-e", "--epochs", required = True, help = "Provide number of epochs to run train net")
ap.add_argument("-i", "--increment", required = True, help = "Provide the increment by which epochs should increase `default is 1`")
ap.add_argument("-s", "--start", required = False, help = "Provide the starting epoch for training")
args = vars(ap.parse_args())

#system arguments
description = args["description"]
model_file = "../models/structure/model" + description + ".json"
#description = model_file.split("/model")[2].split(".json")[0]
epoch_range = int(args["epochs"])
epoch_inc = int(args["increment"])
epoch_start = None
if args["start"] is not None:
    epoch_start = int(args["start"])

data_dict = data_manip.load_data("../preprocessed_data/ppd_exemplar.json")

X_train = np.array(data_dict["X_train"])
X_test = np.array(data_dict["X_test"])
Y_train = np.array(data_dict["Y_train"])
Y_test = np.array(data_dict["Y_test"])

loaded_model = model_manip.load_model(model_file)

loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

if epoch_start != None:
    loaded_model.load_weights("../models/weights/" + description + "/weight" + description + "_" + epoch_start + "_epochs.h5")
else:
    epoch_start = 0

iterations = int(epoch_range/epoch_inc)

for i in range(int(epoch_start)+1, iterations+1):
    print("Training EPOCH: " + str(i))
    loaded_model.fit(X_train, Y_train, batch_size=32, epochs=epoch_inc, validation_data=(X_test, Y_test))

    loaded_model.save_weights("../models/weights/" + description + "/weight" + description + "_" + str(i*epoch_inc) + "_epochs.h5")
