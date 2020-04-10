import json
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import tensorflow_hub as hub

IMAGE_SIZE = 224

# Receive an image path, model pretrained and top_k probabilities and 
# return the actually values of the probabilities along with the class name for every type
def predict(image_path, model_path, top_k, category_path):
    
    print('\n\nLoading Model===>')
    # Load Model and Image
    model = load_model(model_path)    
    img = np.asarray(Image.open(image_path))
    print('Model Loaded.....OK')
    # Image Preprocessing, Convert img to tensor, resize and normalize.\
    print('\nPreprocessing Image ===>')
    img = tf.convert_to_tensor(img)
    img_batch = image_preprocessing(img)    
    print('\nImage Processed.....OK')
    # Prediction
    print('\nPredicting.....')
    probs = model.predict(img_batch)
    probs = probs.flatten()    
    ps, classes = tf.math.top_k(probs, k=top_k, sorted=True)
    ps = ps.numpy()
    classes = classes.numpy()
    classes += 1
    
    if category_path != None:
        c_names = class_names(category_path)
        flower_names = np.array([])
    
    #   Extract class_names of flowers
        for i in classes:
            flower_names = np.append(flower_names, c_names[str(i)])
        classes = flower_names
    
    return ps, classes


def load_model(model_path):
#   Use this line to load Saved_model format
#     model = tf.keras.models.load_model(model_path)
    
#   Use this line to load the h5 model
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
#     model.summary()
    
    return model

# This function resizes the image to (224, 224, 3) and converts it to float32 and normalize the values between 0 and 1
# Also set the image as a batch with only one image
def image_preprocessing(img):
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img_batch = tf.reshape(img, [1, IMAGE_SIZE, IMAGE_SIZE, 3])
    
    return img_batch

# Load Dictionary with Classes names
def class_names(category_path):
    with open(category_path, 'r') as f:
        class_names = json.load(f)
    
    return class_names
