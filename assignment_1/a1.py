import numpy as np
import csv 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import ConfusionMatrixDisplay
import umap
from plotting import plot_scatter, get_confusion_matrix

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

def centre_predictions(string, X, y):
    """
    uses euclidean distance from label means to predict class
    """
    predictions = []
    losses = []
    for idx, x_i in enumerate(X):
        x_i_broadcasted = np.broadcast_to(x_i, means.shape)
        dist = np.linalg.norm(x_i_broadcasted-means)
        pred_y = np.argmin(dist)
        loss = 1 if pred_y == y[idx] else 0
        predictions.append(pred_y)
        losses.append(loss)

    percentage_true_pred = np.sum(losses) / len(losses)
    print (string, f"Percentage of true predictions: {percentage_true_pred:.4f}")

    return predictions, losses

train_centre_pred, _ = centre_predictions("(train)",X, y)

test_centre_pred, _ = centre_predictions("(test)", X_test, y_test)

# ====================================================
### Task 1.4
# ====================================================

neigh = KNeighborsClassifier()
neigh.fit(X,y)

train_pred, test_pred = neigh.predict(X), neigh.predict(X_test)

print(X.shape, train_pred.shape)
train_loss_arr = np.where(train_pred == y, 1, 0)
test_loss_arr = np.where(test_pred == y_test, 1, 0)

def print_pred_perc(string, losses):
    percentage_true_pred = np.sum(losses) / len(losses)
    print (string, f"Percentage of true predictions: {percentage_true_pred:.4f}")

print_pred_perc("(train)", train_loss_arr)
print_pred_perc("(test)", test_loss_arr)

# Confusion matrices

print("Shapes:", len(train_centre_pred), y.shape)

get_confusion_matrix(train_centre_pred, y, "cm_centre_pred_train.png")
get_confusion_matrix(test_centre_pred, y_test, "cm_centre_pred_test.png")
get_confusion_matrix(train_pred, y, "cm_knn_pred_train.png")
get_confusion_matrix(test_pred, y_test, "cm_knn_pred_test.png")

# cm_centre_pred_train = ConfusionMatrixDisplay.from_predictions(train_centre_pred, y)
# cm_centre_pred_test = ConfusionMatrixDisplay.from_predictions(test_centre_pred, y_test)
# cm_knn_pred_train = ConfusionMatrixDisplay.from_predictions(train_pred, y)
# cm_knn_pred_test = ConfusionMatrixDisplay.from_predictions(test_pred, y_test)



