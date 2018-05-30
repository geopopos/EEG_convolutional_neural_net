from keras.models import model_from_json
from keras.utils import plot_model
import sys, json, fileinput
from grnnf import model_manip, data_manip

description = sys.argv[1]
#epoch = sys.argv[2]

model_file = "../models/structure/model" + description + ".json"
loaded_model = model_manip.load_model(model_file)

plot_model(loaded_model, to_file="../models/figures/model" + description + ".png", show_shapes=True)
