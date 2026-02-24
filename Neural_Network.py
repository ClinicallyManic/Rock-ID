#! python

from dotenv import load_dotenv
load_dotenv("removewarnings.env")

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
	img_height = 512   #Sizes are overkilled because I am lazy and its easier then dynamically loading images by alot
	img_width = 512

	#training Dataset reserves 20% of dataset for validation
	dataset_tuple = tf.keras.utils.image_dataset_from_directory(
	path,
	validation_split=0.2,
	subset="both",
	seed=123,
	label_mode="categorical",
	image_size=(img_height, img_width),
	)

  	#returns training and validation sets as a tuple
	return dataset_tuple

train_ds, test_ds = dataset_creation()

#if you want to verify the classes are correctly generating uncomment following 2 lines
#class_names = train_ds.class_names, test_ds.class_names
#for i in range(len(class_names)):
#	if class_names[0][i] != class_names[1][i]:
#		print("break")
#		break
