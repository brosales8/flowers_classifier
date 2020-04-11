# Image Classifier Flowers Dataset 
### Udacity - Machine Learning NanoDegree

**Author:** Bryan Rosales<br>
**Date:** April 8th, 2020


Overview
---

The goal of the project is to create a classifier to correctly predict the class of 102 types of flowers (images) provided by `oxford102-dataset`. The problem was approached using a Neural Network pretrained `MobileNetv2.0 - Transfering Learning` and training it on the new dataset. The dataset was preprocessing to resize images and also data augmentation was required to avoid overfitting due by the number of samples in training dataset.
Please see the following files for more detail:

Files
---
- [Project_Image_Classifier_Project.ipynb](https://github.com/brosales8/flowers_classifier/blob/master/Project_Image_Classifier_Project.ipynb) Jupyter Notebook providing the source code of the classifier, preprocessing and prediction of the images.
- [Project_Image_Classifier_Project.html](https://github.com/brosales8/flowers_classifier/blob/master/Project_Image_Classifier_Project.html) Format HTML required for Udacity submission (Same information like the notebook).
- [predict.py](https://github.com/brosales8/flowers_classifier/blob/master/predict.py) Main file to run the application. Please see Usage section to understand the command line.
- [functions.py](https://github.com/brosales8/flowers_classifier/blob/master/functions.py) Containes useful functions to load image and model, then predict the class name of the image and the probability. Also, there is an additional function to preprocessing the image in order to get the shape expected for the model.
- [label_map.json](https://github.com/brosales8/flowers_classifier/blob/master/label_map.json) File storaging the 102 class names for flowers.
- [/saved_models](https://github.com/brosales8/flowers_classifier/tree/master/saved_model) This folder includes a Tensorflow **SavedModel** format.
- [model_04112020_07_08.zip](https://github.com/brosales8/flowers_classifier/blob/master/model_04112020_07_08.zip) Contains .h5 format version of the model.
<br>Note: 
By default the model uses for the application is .h5 format. However, either one can be used, just uncommented and commented lines in file `functions.py`, function `load_model()`.
<br>For instance:<br>
To load Saved_model format, uncomment `model = tf.keras.models.load_model(model_path)` and comment `model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})` and viceversa for using H5 model. Both models contain the same architecture, only the format is the difference.

-[/test_images]() Folder Contains some images to test the application.


Dataset
---
- [102 Category Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

Usage
---
In order to run the application you should specific the following data in command line:

- *Required field `img_path`, help='Filepath to Image
- *Required field `model_path`, Filepath to Model .H5
- optional: (-k) `top_k` probabilities to return along with the prediction. If not specified, the application returns the most probable class (top1).
- optional: (-c) `category_names` Filepath to class names (JSON file). If not specified, the application returns class number.


Example of shell command:

- Simplest call witout options:<br>
=> python predict.py ./test_images/marigold.jpg ./models/h5/best_model.h5
- including options:<br>
=> python predict.py ./test_images/marigold.jpg ./models/h5/best_model.h5 -c ./label_map.json -k 5 -c ./label_map.json