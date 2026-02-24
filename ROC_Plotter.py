import numpy as np
import sklearn
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# code from copilot, needs to be worked out to see if it works
# function needs to be passed test data set and tensorflow model
def PlotROC(test_ds, model):

    # Extract all labels and images from the test dataset
    y_true = []
    X_test = []

    for images, labels in test_ds:
        X_test.append(images.numpy())
        y_true.append(labels.numpy())

    X_test = np.concatenate(X_test)
    y_true = np.concatenate(y_true)

    # One-hot encode labels
    classes = np.unique(y_true)
    y_true_oh = label_binarize(y_true, classes=classes)
    n_classes = y_true_oh.shape[1]

    y_score = model.predict(X_test)


    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"Class {i} (AUC = {roc_auc[i]:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.show()
