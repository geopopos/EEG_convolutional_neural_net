# Convolutional Neural Networks for EEG Data Categorization

This repository houses a convolutional neural network built using the [Keras machine learning framework](https://keras.io/). The purpose of the network is to decode EEG data. Specifically decode object recognition data collected from users viewing 6 different categories of images (human body, human face, animal body, animal face, inanimate natural object, and inanimate man-made object). a test bank of 72 images were displayed to users. The convolutional neural network attempts to determine which category of image a use was looking at based on the EEG data collected. The image bank can be seen below. Please visit the stanford research article that conducted data collection [here](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0135697) to find out more about procedure.

<p align="center">
  <img src="http://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0135697.g001" width="350"/>
</p>

The convolutional neural network is based off of a net created from Filter Bank Common Spatial Patterns the original neural network can be found [here](https://arxiv.org/abs/1703.05051)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

There are a few prerequisites needed before running this project on your machine
- First, you must have a copy of python 3.5.0 or higher and pip installed on your machine
- Next, you will need the python libraries listed below
```
numpy==1.14.0
Keras==2.1.3
scipy==0.18.1
matplotlib==1.5.3
```
- Lastly, you will need a copy of the [preprocessed eeg data](https://www.dropbox.com/s/9udeedmagnanf4n/preprocessed_data.zip?dl=0) used for this convolutional neural network.

### Installing

Assuming python 3 is already installed you will need to install pip on your machine.

You can do so by running the following command on your linux machine
```
apt install python3-pip
```

Next, install numpy, scipy, and matplotlib
```
pip install numpy
pip install scipy
pip install matplotlib
```

Keras installation is quite lengthy so I will refer you to the [Documentation](https://keras.io/#installation) for your reading.

Now download the [preprocessed data](https://www.dropbox.com/s/9udeedmagnanf4n/preprocessed_data.zip?dl=0)
Extract the ```preprocessed_data.zip file```
Store the ```preprocessed_data/``` folder in the root directory of the project

## Project Layout
```
|-- data_dumps/
|-- graphs/
|-- models/
|   |-- figures/
|   |-- structure/
|   `-- weights/
|-- preprocessed_data/
`-- scripts/
    `-- grnnf/
```
The project is set up in 5 main folders
The first folder ```data_dumps/``` stores the accuracy and loss for the training and test for a set amount of epochs. This data is later used to examine peak accuracies, losses, overfitting, etc.

The next folder ```graphs/``` contains images of graphs of accuracies and losses using matplotlib

The ```models/``` folder has 3 subfolders ```figures/```, ```structure/```, and ```weights/```. The figures folder contains images of the model layout. the structure folder contains the JSON files related to each different models structure, and the weights folder contains weights for each model structure after being trained a certain number of epochs.

The ```preprocessed_data/``` folder contains the EEG data recorded at Stanford University.

The ```scripts/``` folder contains all of the python scripts that compile train and test the neural network along with auxillary scripts that plot accuracies.

## Running the tests

once all of the prerequisites are donwloaded and installed its time to train and test the neural network on our data

1. make sure you are in the scripts directory
```
cd scripts/
```

2. we're going to want to construct our neural network model. the ```shallowConvNet.py``` script will do the job if you want to create a different model simply edit the ```shallowConvNet.py```. To run the script use python and enter the description for this net
```
python3 shallowConvNet.py -d [description for model]
```
This will create the model and store it in the ```models/structure/``` folder

3. Next let's train the model for 20 epochs.
```
python3 runModelEpochs.py -d source_model21 -e 20 -i 1
```
The models weights for each training epoch are stored in the ```models/weights/``` folder

4. to get train and test accuracies and losses for epoch we will run the ````runTesting.py``` script
this script will take in the weights for the specified range (this is zero based and upperbound exclusive) so this script will check accuracies for the 20 weights files we stored earlier. If the accuracies and losses are already calculated the script will simply close.
```
python3 runTesting.py -d source_model -s 0 -e 20
```

5. To plot the loss or accuracies for the net you just trained the ``` plotthejson.py``` script is used
```
python3 plotthejson.py -d source_model -a acc -s 0 -e 20
```
that will result in a graph similar to this:
<p align="center">
  <img src="https://drive.google.com/open?id=1ZjXls8SKn0wwbiauLqenhH5rYFVASFe4" width="350"/>
</p>
## Authors

* **Georgios Basilios Roros** - *Initial work* - [George Roros](https://github.com/geopopos)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
