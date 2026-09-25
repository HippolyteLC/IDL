import numpy as np
import csv 

X = np.loadtxt(r'/local/s4099699/IDL/assignment_1/data/train_in - Copy.csv', delimiter=',')
y = np.loadtxt(r'/local/s4099699/IDL/assignment_1/data/train_out - Copy.csv', delimiter=',').ravel()

unique_labels = np.unique(y)
means = np.array([X[y == label].mean(axis=0) for label in unique_labels])

# dist_matrix = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=-1)
# np.set_printoptions(precision=2, suppress=True, linewidth=100)
# print(dist_matrix)
n_labels = unique_labels.shape[0]
dist_matrix = []
for i in range(n_labels):
    dist_row = []
    for j in range(n_labels):
        euclidean_dist = np.linalg.norm(means[i]-means[j])
        dist_row.append(euclidean_dist)
    dist_matrix.append(dist_row)
dist_matrix = np.array(dist_matrix)
np.set_printoptions(precision=2, suppress=True, linewidth=100)

    
print(dist_matrix)

