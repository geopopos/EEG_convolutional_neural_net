# Convolutional Neural Networks for EEG Data Categorization

This repository houses a convolutional neural network built using the [Keras machine learning framework](https://keras.io/). The purpose of the network is to decode EEG data. Specifically decode object recognition data collected from users viewing 6 different categories of images (human body, human face, animal body, animal face, inanimate natural object, and inanimate man-made object). a test bank of 72 images were displayed to users. The convolutional neural network attempts to determine which category of image a use was looking at based on the EEG data collected. The image bank can be seen below. to find out more about data collection and procedure please visit the page for the [original research](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0135697)

<p align="center">
  <img src="http://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0135697.g001" width="350"/>
</p>

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

###Prerequisites

There are a few prerequisites needed before running this project on your machine
```
numpy==1.14.0
Keras==2.1.3
scipy==0.18.1
matplotlib==1.5.3
```