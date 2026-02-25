import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub
import Neural_Network as NN

#Sub-function of bargraph for counting inside of labels
def count(dataset):
	names = dataset.class_names
	name_count = len(names)
	count = {}
	#makes dictionary of names with a value of 0
	for z in range(name_count):
		count[names[z]] = 0

	#image is unused do not change this
	#this loop counts increments the value of a key
	dataset = dataset.unbatch()
	for image, label in dataset:
		for i in range(name_count):
			if label[i] == 1:
				count[names[i]]+=1
	return count

#Trying to make a bar graph detailing classes in testing and training set to demonstrate imbalance
def bar_graph(train,test):
	train_counted = count(train)
	test_counted = count(test)

count(NN.train_ds)
#bar_graph(NN.train_ds, NN.test_ds)
