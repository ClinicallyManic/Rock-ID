#! python

#if packages cannot resolve please remember to install the packages using pip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub

#TODO
#Change seed to actually be random for final version set seed is fine for small scale testing
#Move dataset creation and management to independant function
#Make neural network to process dataset confer with group for details
#Remove commented code for final version


# Download latest version also concatenate \\rocks to the end to end up in the correct directory
path = kagglehub.dataset_download("neelgajare/rocks-dataset")
path+= "\\rocks"
#print("Path to dataset files:", path) #commented out for brevity maintained for so we can check

#code nicked from google gemini because the relavant kaggle tutorial is bad
batch_size = 32
img_height = 512   #Sizes are overkilled because I am lazy and its easier then dynamically loading images by alot
img_width = 512

#training Dataset reserves 20% of dataset for validation
train_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

#validation dataset uses 20% of dataset to make sure we train right
val_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

#if you want to verify the classes are correctly generating
#class_names = train_ds.class_names
#print("Found these classes:", class_names)