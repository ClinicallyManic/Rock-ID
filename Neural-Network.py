#! python

#if packages cannot resolve please remember to install the packages using pip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub

#TODO
#Change seed to actually be random for final version set seed is fine for small scale testing
#Make neural network to process dataset confer with group for details
#Remove commented code for final version

#Function creates the training and validation datasets and returns aforementioned datasets
def dataset_creation():
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

  #returns training and validation sets
  return train_ds, val_ds

dataset_creation()

#if you want to verify the classes are correctly generating uncomment following 2 lines
#class_names = train_ds.class_names
#print("Found these classes:", class_names)