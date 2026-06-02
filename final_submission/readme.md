# Semantic Segmentation for Cat Image Mask Prediction Using Deep Learning

## 1. Project Description

This project is a deep learning-based semantic segmentation project. The main aim of the project is to train a model that can predict a binary segmentation mask from an input image.

Semantic segmentation is a computer vision task where each pixel in an image is classified into a specific category. In this project, the model tries to separate the cat object from the background. The cat region is represented with white pixels, and the background is represented with black pixels.

The project was implemented using Python and PyTorch. The model was trained on image-mask pairs and tested by comparing the predicted mask with the real mask.

---

## 2. File Descriptions

The project folder includes the following files:

### `dataset.py`
This file contains the dataset class. It loads the input images and their corresponding segmentation masks. It also applies preprocessing operations such as resizing and converting images into tensors.

### `model.py`
This file contains the deep learning model architecture. The model is based on convolutional neural network layers and is designed for binary semantic segmentation.

### `train.py`
This file is used to train the segmentation model. It loads the dataset, trains the model for a number of epochs, calculates the training loss, evaluates validation pixel accuracy, and saves the trained model.

### `predict.py`
This file is used to test the trained model on a sample image. It loads the saved model and creates a predicted segmentation mask.

### `utils.py`
This file contains helper functions used in the project.

### `test_dataset.py`
This file is used to test whether the dataset loading process works correctly.

### `test_model.py`
This file is used to test whether the model architecture works correctly.

### `setup.py`
This file is used for project setup and package-related configuration.

### `segmentation_model.pth`
This is the saved trained model file.

### `training_loss.png`
This image shows the training loss curve. The loss decreased during training, which means the model learned from the training data.

### `validation_accuracy.png`
This image shows the validation pixel accuracy curve. The validation accuracy increased during training, which shows that the model improved on unseen validation images.

### `prediction_result.png`
This image shows the original image, the real segmentation mask, and the predicted segmentation mask.

---

## 3. Dataset Description

The dataset used in this project consists of image and mask pairs.

Each sample contains:

- An original RGB image
- A binary segmentation mask

In the binary mask:

- White pixels represent the cat object
- Black pixels represent the background

The images were resized to 128x128 pixels in order to reduce computational cost and make the training process faster on CPU.

If the dataset is not included in the submission folder because of file size limitations, it can be accessed from the following link:

Dataset link: ADD_DATASET_LINK_HERE

---

## 4. Required Libraries

The project requires the following libraries:

```text
Python 3.x
torch
torchvision
Pillow
matplotlib
numpy