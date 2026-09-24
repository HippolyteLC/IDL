import numpy as np
import csv 

X = np.loadtxt(r'/local/s4099699/IDL/assignment_1/data/train_in - Copy.csv', delimiter=',')
y = np.loadtxt(r'/local/s4099699/IDL/assignment_1/data/train_out - Copy.csv', delimiter=',').ravel()

unique_labels = np.unique(y)
means = np.array([X[y == label].mean(axis=0) for label in unique_labels])



