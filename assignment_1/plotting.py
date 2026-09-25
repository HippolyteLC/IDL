import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(X, y, filename):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
    plt.savefig(filename)
    plt.close()