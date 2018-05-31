# Convolutional Neural Networks for EEG Data Categorization

This repository houses a convolutional neural network built using the [Keras machine learning framework](https://keras.io/). The purpose of the network is to decode EEG data. Specifically decode object recognition data collected from users viewing 6 different categories of images (human body, human face, animal body, animal face, inanimate natural object, and inanimate man-made object). a test bank of 72 images were displayed to users. The convolutional neural network attempts to determine which category of image a use was looking at based on the EEG data collected. The image bank can be seen below. to find out more about data collection and procedure please visit the page for the [original research](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0135697)

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


## Authors

* **Georgios Basilios Roros** - *Initial work* - [Geopopos](https://github.com/geopopos)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
