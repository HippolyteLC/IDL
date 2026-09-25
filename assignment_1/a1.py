import numpy as np
import csv 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import umap
from plotting import plot_scatter



X = np.loadtxt(r'/local/s4099699/IDL/assignment_1/data/train_in - Copy.csv', delimiter=',')
y = np.loadtxt(r'/local/s4099699/IDL/assignment_1/data/train_out - Copy.csv', delimiter=',').ravel()
X_test = np.loadtxt(r'/local/s4099699/IDL/assignment_1/data/test_in - Copy.csv', delimiter=',')
y_test = np.loadtxt(r'/local/s4099699/IDL/assignment_1/data/test_out - Copy.csv', delimiter=',')


# ====================================================
### Task 1.1 
# ====================================================

unique_labels = np.unique(y)
means = np.array([X[y == label].mean(axis=0) for label in unique_labels])

n_labels = unique_labels.shape[0]
dist_matrix = []
for i in range(n_labels):
    dist_row = []
    for j in range(n_labels):
        euclidean_dist = np.linalg.norm(means[i]-means[j])
        dist_row.append(euclidean_dist)
    dist_matrix.append(dist_row)
dist_matrix = np.array(dist_matrix)
np.set_printoptions(precision=2, suppress=True, linewidth=100) # Comment out to print normally
# print(dist_matrix) # Uncomment to check matrix


# ====================================================
### Task 1.2
# ====================================================

# scaled_data ? 

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# print(X_pca.shape)
# plot_scatter(X_pca, y, "pca_plot.png")

# X_tsne = TSNE(n_components=2).fit_transform(X)
# print(X_tsne.shape)
# plot_scatter(X_tsne, y, "tsne_plot.png")

# reducer = umap.UMAP()
# X_umap = reducer.fit_transform(X)
# print(X_umap.shape)
# plot_scatter(X_umap, y, "umap_plot.png")

# # plot PCA centres
# X_pca_centres = pca.fit_transform(means)
# plot_scatter(X_pca_centres, unique_labels, "pca_centres_plot.png")

# ====================================================
### Task 1.3
# ====================================================

def centre_predictions(X, y):
    """
    uses euclidean distance from label means to predict class
    """
    predictions = []
    losses = []
    for idx, x_i in enumerate(X):
        x_i_broadcasted = np.broadcast_to(x_i, means.shape)
        dist = np.linalg.norm(x_i_broadcasted-means)
        pred_y = np.argmin(dist)
        loss = 0 if pred_y == y[idx] else 1
        predictions.append(pred_y)
        losses.append(loss)

    percentage_true_pred = (len(losses) - np.sum(losses)) / len(losses)
    print (f"Percentage of true predictions (train): {percentage_true_pred:.4f}")

    return predictions, losses

centre_predictions(X, y)

centre_predictions(X_test, y_test)


# ====================================================
### Task 1.4
# ====================================================

neigh = KNeighborsClassifier()
neigh.fit(X,y)

train_loss, test_loss = [], []
for x_i_train, x_i_test in zip(X, X_test):
    train_pred, test_pred = 

