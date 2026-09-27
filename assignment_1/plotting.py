import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_plot_scatter(X, y, filename, title):
    scatt = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
    plt.legend(*scatt.legend_elements(), title="Label")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

    
def save_confusion_matrix(pred, y, filename, title):
    ConfusionMatrixDisplay.from_predictions(pred, y)
    plt.title(title)
    plt.savefig(filename)
    plt.close()
