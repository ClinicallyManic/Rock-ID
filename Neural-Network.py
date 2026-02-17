#! python

#if packages cannot resolve please remember to install the packages using pip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub

# Download latest version
path = kagglehub.dataset_download("neelgajare/rocks-dataset")
print("Path to dataset files:", path)

#code nicked from google gemini because the relavant kaggle tutorial is bad
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

<<<<<<< HEAD
=======
print("Path to dataset files:", path)
>>>>>>> 5ff0ce4cd96ddcfa16aa83def6924e61fd0ee512
