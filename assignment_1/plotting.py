import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(X, y, filename):
    scatt = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
    plt.legend(*scatt.legend_elements(), title="Label")
    plt.savefig(filename)
    plt.close()