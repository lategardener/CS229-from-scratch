import numpy as np
import seaborn as sns

from LogisticRegression.src.classification import confusion_elements

def confusion_matrix(X, y, theta):
    TP, TN, FP, FN = confusion_elements(X, y, theta)
    cf_matrix = np.array([[TP, FN], [FP, TN]])

    group_names = ["True Pos","False Neg","False Pos","True Neg"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')