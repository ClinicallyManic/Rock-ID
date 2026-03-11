import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub
import Neural_Network as NN

def display_chart(gini, kappa, f1):
    fig, ax = plt.subplots()
    scores = ['gini', 'kappa', 'f1']
    counts = [gini, kappa, f1]
    bar_colors = ['tab:green', 'tab:orange', 'tab:red']
    ax.bar(scores, counts, color=bar_colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy scores of measuring methods over 42 epochs')
    plt.show()

def extract_labels(dataset):
    labels = []
    for images, lbls in dataset:
        labels.extend(lbls.numpy())  
    return np.array(labels)

def gini_impurity(y):
    classes = np.unique(y)
    n_samples = len(y)
    gini = 1.0

    for cls in classes:
        p = np.sum(y == cls) / n_samples
        gini -= p ** 2

    return gini

# scores[0] = a = TP, scores[1] = b = FP, scores[2] = c = FN, scores[3] = d = TN
def get_abcd(y_true, ver_true):
    classes = np.unique(np.concatenate([y_true, ver_true]))
    result = [0, 0, 0, 0]
    for cls in classes:
        result[0] += np.sum((y_true == cls) & (ver_true == cls))
        result[1] += np.sum((y_true == cls) & (ver_true != cls))
        result[2] += np.sum((y_true != cls) & (ver_true == cls))
        result[3] += np.sum((y_true != cls) & (ver_true != cls))
    print(result)
    return result

def kappa_score(y, verification):
    n_samples = len(y)
    n_ver = len(verification)
    scores = get_abcd(y, verification)
    po = (scores[0] + scores[3]) / n_samples
    pc = ((scores[0] + scores[1]) / n_samples) * ((scores[0] + scores[2]) / n_samples)
    pi = ((scores[2] + scores[3]) / n_samples) * ((scores[1] + scores[3]) / n_samples)
    pe = pc + pi
    kappa = (po - pe)/(1.0 - pe)
    return kappa
    
    

def f1_score(y, verification):
    n_samples = len(y)
    scores = get_abcd(y, verification)
    return (2 * scores[0]) / ((2*scores[0]) + scores[1] + scores[2])


# train_ds, test_ds = NN.dataset_creation()
test_ds = NN.test_ds
y_true = extract_labels(NN.test_ds)

y_pred_probs = NN.probability_model.predict(NN.test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

gini = gini_impurity(test_ds)
kappa = kappa_score(y_pred, y_true)
f1 = f1_score(y_pred, y_true)
display_chart(gini, kappa, f1)

