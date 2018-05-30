import sys, random, time, json, os

import numpy as np

from scipy.io import loadmat

from keras.models import Sequential, model_from_json

from keras.layers import Dense, Activation, Flatten

from keras.layers import Convolution1D, Convolution2D, Convolution3D, AveragePooling2D, Reshape, Lambda, Permute, Dropout

from keras.utils import np_utils

from keras import backend as K

from grnnf import parse_data, model_manip, data_manip

description = ""
if sys.argv[2] != None:
    description = sys.argv[2]
else:
    description = str(time.time())

"""data_dict = data_manip.load_data("../preprocessed_data/ppd.json") #parse_data.parse_data("../../../data/")
X_train = np.array(data_dict["X_train"])
Y_train = np.array(data_dict["Y_train"])
X_test = np.array(data_dict["X_test"])
Y_test = np.array(data_dict["Y_test"])
"""
model = Sequential()

def logIt(x):
    return K.log(x)

model.add(Dropout(0.4, input_shape=(1,124,32)))
model.add(Convolution2D(40, (1,10), activation="relu", data_format="channels_first"))
model.add(Permute((3,2,1)))
model.add(Reshape((1,23,124,40)))
model.add(Dropout(0.4))
model.add(Convolution3D(40, (1,124,40), activation="relu", data_format="channels_first"))
model.add(Dropout(0.4))
model.add(Lambda(lambda x: x ** 2))
model.add(Reshape((40,23,1)))
model.add(Dropout(0.4))
#model.add(AveragePooling2D((1, 10), (1, 2)))
#model.add(Activation("relu"))
#literal garbage
#model.add(Lambda(logIt))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(6, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#save model structure as json to 'model/structure/'
model_manip.save_model("../models/structure/model" + description + ".json", model)

directory = "../models/weights/" + description + "/"
if not os.path.exists(directory):
        os.makedirs(directory)

"""
def modelExecute(epochs):
#    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_data=(X_test, Y_test))

    test_score = model.evaluate(X_test, Y_test, verbose=0)

    return history.history

epochs = int(sys.argv[1])
scores = modelExecute(epochs)

epochs_array = list(range(1,epochs+1))
train_loss_array = scores["loss"]
test_loss_array = scores["val_loss"]
train_accuracy_array = scores["acc"]
test_accuracy_array = scores["val_acc"]

#save train/test loss and accuracy as json to 'data_dump/'
with open("../data_dumps/training-test-history" + description + ".json", "w") as outputscore:
    json.dump(scores, outputscore)

#save model structure as json to 'model/structure/'
#model_manip.save_model("../models/structure/model" + description + ".json", model)

#save model weights as hdf5 to 'model/weights/'
model.save_weights("../models/weights/model" + description + ".h5")
"""
