#! python

from dotenv import load_dotenv
load_dotenv("removewarnings.env")

#if packages cannot resolve please remember to install the packages using pip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub
import os.path
import random as r
import argparse

import ROC_Plotter as ROC

parser = argparse.ArgumentParser(
					prog='Neural Network',
					description='A neural network that analizes rock images')

parser.add_argument("-r", "--retrain", action="store_true")

args = parser.parse_args()

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
	seed=6967,
	image_size=(img_height, img_width),
	)

  	#returns training and validation sets as a tuple
	return dataset_tuple

train_ds, test_ds = dataset_creation()

model = None

if not os.path.exists("./model/"):
	os.makedirs("./model/")

if os.path.exists("./model/model.keras") and not args.retrain:
	model = tf.keras.models.load_model("./model/model.keras")
else:
	#RandomContrast must go before flatten
	model = tf.keras.Sequential([
		tf.keras.layers.RandomContrast(factor=0.5),
		tf.keras.layers.Flatten(input_shape=(512, 512, 3)),
		tf.keras.layers.Rescaling(1./255),
	    tf.keras.layers.Dense(128),
		tf.keras.layers.Dense(108),
		tf.keras.layers.Dense(72),
		tf.keras.layers.Dense(64),
		tf.keras.layers.Dense(53)
	])

	model.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'],
				  )

	checkpoint = tf.keras.callbacks.ModelCheckpoint("./model/model.keras", save_best_only = True, monitor='accuracy', mode='max', verbose=1)

	model.fit(train_ds, epochs=10, callbacks=[checkpoint])

probability_model = tf.keras.Sequential([model,
                                        tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_ds)

model.summary()
model.get_metrics_result()

# ROC.PlotROC(test_ds, model)
