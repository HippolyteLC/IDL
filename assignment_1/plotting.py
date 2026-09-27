import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_plot_scatter(X, y, filename):
    scatt = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
    plt.legend(*scatt.legend_elements(), title="Label")
    plt.savefig(filename)
    plt.close()

    
def save_confusion_matrix(pred, y, filename):
    ConfusionMatrixDisplay.from_predictions(pred, y)
    plt.savefig(filename)
    plt.close()
