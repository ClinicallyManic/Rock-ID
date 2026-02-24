import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub

def display_chart(gini, kappa, f1, epochs):
    fig, ax = plt.subplots()
    scores = ['gini', 'kappa', 'f1']
    counts = [gini, kappa, f1]
    bar_colors = ['tab:green', 'tab:orange', 'tab:red']
    ax.bar(scores, counts, color=bar_colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy scores of measuring methods over ' + epochs + ' epochs')
    plt.show()


def gini_impurity(y):
    classes = np.unique(y)
    n_samples = len(y)
    gini = 1.0

    for cls in classes:
        p = np.sum(y == cls) / n_samples
        gini -= p ** 2

    return gini

# scores[0] = a = TP, scores[1] = b = FP, scores[2] = c = FN, scores[3] = d = TN
def get_abcd(y, verification):
    classes = np.unique(y)
    control = np.unique(verification)
    n_samples = len(y)
    result = [0, 0, 0, 0]
    for ii in range(0, n_samples):
        p = np.sum(y == classes[ii])
        q = np.sum(verification == control[ii])
        pnot = n_samples - p
        qnot = n_samples - q
        
        dif_positive = abs(q - p)
        dif_negative = abs(qnot - pnot)

        if p < q:
            result[0] += p
        else:
            result[0] += q

        if pnot < qnot:
            result[3] += pnot
        else:
            result[3] += qnot
        result[1] += dif_positive
        result[2] += dif_negative

    return result

def kappa_score(y, verification):
    n_samples = len(y)
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
