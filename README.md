# Facial-Landmark-Detection-Using-Pytorch

## Problem Statement
This project will be all about defining and training a **Convolutional Neural Network** to perform facial keypoint detection, and using **Computer Vision** techniques to transform images of faces.

**Facial keypoints** (also called facial landmarks) are the small magenta dots shown on the faces in the image below. In each training and test image, there is a single face and **68 keypoints, with coordinates (x, y), for that face**.  These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.

<img src='images/landmarks_numbered.jpg' width=30% height=30%/>

## Load and Visualize Data
The first step in working with any dataset is to become familiar with your data; you'll need to load in the images of faces and their keypoints and visualize them! This set of image data has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

#### Training and Testing Data
This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

* 3462 of these images are training images, for you to use as you create a model to predict keypoints.
* 2308 are test images, which will be used to test the accuracy of your model.

The information about the images and keypoints in this dataset are summarized in CSV files, which we can read in using `pandas`.

## Data Preprocessing and Data Transforms
All the images are not of the same size, and neural networks often expect images that are standardized; a fixed size, with a normalized range for color ranges and coordinates, and (for PyTorch) converted from numpy lists and arrays to Tensors.

Therefore, the following preprocessing steps are performed:

-  ``Normalize``: to convert a color image to grayscale values with a range of [0,1] and normalize the keypoints to be in a range of about [-1, 1]
-  ``Rescale``: to rescale an image to a desired size.
-  ``RandomCrop``: to crop an image randomly.
-  ``ToTensor``: to convert numpy images to torch images.
-  ``Resize``: to resize an image to a desired size.

## Model Building and Training : 
### Using the Pytorch Convolutional Neural Network
In the `models.py` file, we have defined our neural network using **``pytorch``** and the following steps are performed:
1. Define a CNN with images as input and keypoints as output
2. Construct the transformed FaceKeypointsDataset
3. Train the CNN on the training data, tracking loss
4. See how the trained model performs on test data
5. If necessary, modify the CNN structure and model hyperparameters, so that it performs well
6. Save the best model locally.

## Model Prediction
The following steps are performed here:
1. Loading the previous saved model(best model) from training phase.
2. Using test images and predicting the facial keypoints/landmarks on the test images.

## Evaluation and Visualization
1. loss function used is ``mean squared error``.
2. optimizer used is ``Adam``.
3. Visualization is done using ``matplotlib`` and ``OpenCv`` libraries.
