from keras.models import Sequential, model_from_json

from keras.layers import Dense, Activation, Flatten, Convolution2D, Convolution3D, AveragePooling2D, Reshape, Lambda, Permute

from keras.utils import np_utils

from keras import backend as K

import numpy as np

import sys, json, fileinput, argparse

from os import listdir

from os.path import isfile, join

from grnnf import model_manip, data_manip

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--description", required = True, help = "Provide a descriptor for this new neural network structure")
ap.add_argument("-s", "--start", required = False, help = "Provide the starting epoch for testing")
ap.add_argument("-e", "--end", required = False, help = "Provide the ending epoch for testing")
args = vars(ap.parse_args())

description = args["description"]
start_epoch = None
end_epoch = None
if args["start"] is not None:
    start_epoch = int(args["start"])
if args["end"] is not None:
    end_epoch = int(args["end"])

results = {"acc":[], "loss":[], "val_acc":[], "val_loss":[]}

#load model structure from json
model_file = "../models/structure/model" + description + ".json"
loaded_model = model_manip.load_model(model_file)

#load weights into loaded model
weights_path = "../models/weights/" + description + "/"
weights_files = [f for f in listdir(weights_path) if isfile(join(weights_path, f))]

#store data_file name for later access
data_file = "../data_dumps/training-test-history-" + description + ".json"

n = len(weights_files)

for i in range(n):
    for j in range(n-i-1):
        if int(weights_files[j].split(description + "_")[1].split("_")[0]) > int(weights_files[j+1].split(description + "_")[1].split("_")[0]):
            weights_files[j], weights_files[j+1] = weights_files[j+1], weights_files[j]

#check the which epochs results have been processed for and which epochs weights exist for
epoch_weights = weights_files[-1].split(description + "_")[1].split("_")[0]
epoch_results = 0
try:
    with open(data_file, "r") as f:
        results = json.load(f)

    epoch_results = results["acc"][-1][0]
except:
    print("no file exists for current models results")
    
if epoch_weights == epoch_results:
    print("all scores have been recorded for existing weights files")
    sys.exit()


#load preprocessed data
data_dict = data_manip.load_data("../preprocessed_data/ppd_exemplar.json")

X_train = np.array(data_dict["X_train"])
Y_train = np.array(data_dict["Y_train"])
X_test = np.array(data_dict["X_test"])
Y_test = np.array(data_dict["Y_test"])

if start_epoch != None:
    weights_files = weights_files[start_epoch:end_epoch]

#compile model for running
loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

for weights in weights_files:
    epoch = int(weights.split(description + "_")[1].split("_")[0])
    
    if epoch < epoch_results+1:
        continue

    print("Epoch: " + str(epoch))
    loaded_model.load_weights(weights_path + weights)

    train_score = loaded_model.evaluate(X_train, Y_train, verbose=1)
    test_score = loaded_model.evaluate(X_test, Y_test, verbose=1)

    results["loss"].append([epoch, train_score[0]])
    results["acc"].append([epoch, train_score[1]])
    results["val_loss"].append([epoch, test_score[0]])
    results["val_acc"].append([epoch, test_score[1]])

#save train/test loss and accuracy as json to 'data_dump/'
with open(data_file, "w") as outputscore:
        json.dump(results, outputscore)
