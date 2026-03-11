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

def Plot_AccurateVInacc(dataset):
    test_labels = dataset.class_names
    test_totals = count(dataset)
    
    #Get true positives
    y_true = []

    for _, labels in dataset:
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)

    logits = NN.model.predict(dataset)
    y_pred = np.argmax(logits, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    print(np.sum(tp)/np.sum(test_totals) * 100)
    #Graphing
    fig, ax = plt.subplots()
    fig.set_size_inches(10,7)
    ax.barh(test_labels, (tp/test_totals) * 100, label='TP', color='skyblue')
    ax.barh(test_labels, ((test_totals - tp)/test_totals) * 100, label='Total-TP', left=(tp/test_totals) * 100, color='purple')
    ax.set_xlabel('%')
    ax.set_ylabel('Class')
    ax.set_xticks(np.linspace(0, 100, 11))
    ax.set_title("Accurate Predictions VS. Inaccurate Predictions")
    ax.legend(bbox_to_anchor=(1,0.95))
    ax.margins(x=0)
    plt.margins(y=0.01)
    plt.tight_layout()
    plt.show()

Plot_AccurateVInacc(NN.test_ds)