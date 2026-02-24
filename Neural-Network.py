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

train_ds, test_ds = dataset_creation()

import matplotlib.pyplot as plt
import tensorflow as tf

#Get the class names (so we see "cat" instead of "0")
class_names = train_ds.class_names

#Take a single batch from the dataset
#.take(1) ensures we don't try to loop through the entire dataset
for images, labels in train_ds.take(1):
    
    #Create a figure (10x10 inches)
    plt.figure(figsize=(10, 10))
    
    #Loop through the first 9 images in the batch
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        
        # Display the image
        # .numpy().astype("uint8") converts the tensor values (float) back to integers (0-255)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        # Display the label
        # labels[i] is an integer (e.g., 0), class_names converts it to string (e.g., "cat")
        plt.title(class_names[labels[i]])
        
        # Turn off axis coordinates for a cleaner look
        plt.axis("off")

# Show the plot
plt.show()


#if you want to verify the classes are correctly generating uncomment following 2 lines
#class_names = train_ds.class_names
#print("Found these classes:", class_names)