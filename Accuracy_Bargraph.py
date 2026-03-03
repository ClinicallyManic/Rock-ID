import numpy as np
from sklearn.calibration import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub
import Neural_Network as NN
#import Bargraph_Visual as BV
def count(dataset):
    names = dataset.class_names
    name_count = len(names)
    counts = tf.zeros([name_count], dtype=tf.int32)
    for images, labels in dataset:
        batch_counts = tf.math.bincount(labels, minlength = name_count)
        counts += batch_counts 
    return counts

def Plot_AccurateVTotal():
    test_labels = NN.test_ds.class_names
    test_totals = count(NN.test_ds)
    
    #Get true positives
    y_true = []

    for _, labels in NN.test_ds:
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)

    logits = NN.model.predict(NN.test_ds)
    y_pred = np.argmax(logits, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    #Graphing
    fig, ax = plt.subplots()
    fig.set_size_inches(8,7)
    ax.barh(test_labels, test_totals - tp, label='Total - TP', color='purple')
    ax.barh(test_labels, tp, label='TP', left=test_totals - tp, color='skyblue')
    ax.set_xlabel('Number of images in class')
    ax.set_ylabel('Class')
    ax.set_title("Accurate Predictions VS. Total")
    ax.legend()
    ax.set_xlim(0, 20)
    ax.set_xticks(np.linspace(0, 20, 11))
    plt.tight_layout()
    plt.show()

Plot_AccurateVTotal()