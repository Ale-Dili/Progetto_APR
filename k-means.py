import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import seaborn as sns


X_train = np.load("data/processed_data_features/X_train.npy")
X_val   = np.load("data/processed_data_features/X_val.npy")
X_test  = np.load("data/processed_data_features/X_test.npy")

y_train = np.load("data/processed_data_features/y_train.npy")
y_val   = np.load("data/processed_data_features/y_val.npy")
y_test  = np.load("data/processed_data_features/y_test.npy")


X = np.concatenate((X_train, X_val, X_test), axis=0)
y = np.concatenate((y_train, y_val, y_test), axis=0)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k = 7
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')


sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters, cmap='viridis', alpha=0.7)
ax1.set_title('Cluster (k-means)')
ax1.set_xlabel('PCA 1')
ax1.set_ylabel('PCA 2')
ax1.set_zlabel('PCA 3')
fig.colorbar(sc1, ax=ax1, label='Cluster')


sc2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='tab10', alpha=0.7)
ax2.set_title('True Labels')
ax2.set_xlabel('PCA 1')
ax2.set_ylabel('PCA 2')
ax2.set_zlabel('PCA 3')
fig.colorbar(sc2, ax=ax2, label='Label')


def on_move(event):
    if event.inaxes in [ax1, ax2]:
        azim, elev = event.inaxes.azim, event.inaxes.elev
        ax1.view_init(elev, azim)
        ax2.view_init(elev, azim)
        fig.canvas.draw_idle()


fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.tight_layout()
plt.show()


