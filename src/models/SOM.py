import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Union


class SOM:
    def __init__(
        self, 
        m: int, 
        n: int, 
        dim: int,
        weight_init_method: Union[str, callable] = "uniform",
        grid_metric: str = "euclid",
        neighborhood_kernel: str = "gaussian",
        lr_schedule: callable = None,
        radius_schedule: callable = None,
        seed=None
    ):
        
        if m <= 0 or n <= 0 or dim <= 0:
            raise ValueError("SOM dimensions must be positive integers")
        self.m, self.n = m, n
        self.dim = dim

        if seed:
            np.random.seed(seed)

        self.weight_init_method = weight_init_method
        self.weights = None

        self.grid_metric = grid_metric
        self.neighborhood_kernel = neighborhood_kernel

        self.lr_schedule = lr_schedule or (lambda t, T: 0.1 * (1 - t / T))
        self.radius_schedule = radius_schedule or (
            lambda t, T: max(m, n) / 2 * (1 - t / T)
        )

        self.q_error = []
        self.avg_adjust = []

    def init_weights(self, data):
        if callable(self.weight_init_method):
            w = self.weight_init_method(data, self.m, self.n, self.dim)

        elif self.weight_init_method == "uniform":
            w = np.random.rand(self.m, self.n, self.dim)

        elif self.weight_init_method == "data_range":
            lo = data.min(axis=0)  # shape (dim,)
            hi = data.max(axis=0)  # shape (dim,)
            w = (hi - lo) * np.random.rand(self.m, self.n, self.dim) + lo

        elif self.weight_init_method == "sample":
            idx = np.random.choice(data.shape[0], self.m * self.n, replace=True)
            w = data[idx].reshape(self.m, self.n, self.dim)

        elif self.weight_init_method == "pca":
            pca = PCA(n_components=self.dim).fit(data)
            coords = np.indices((self.m, self.n)).transpose(1, 2, 0).astype(float)
            coords[..., 0] /= self.m - 1
            coords[..., 1] /= self.n - 1
            pc_grid = coords.reshape(-1, 2) * np.sqrt(pca.explained_variance_)
            w = pc_grid @ pca.components_
            w += pca.mean_  # broadcast add
            w = w.reshape(self.m, self.n, self.dim)
        elif self.weight_init_method == "kmeans":
            km = KMeans(n_clusters=self.m * self.n, n_init=10).fit(data)
            w = km.cluster_centers_.reshape(self.m, self.n, self.dim)

        else:
            raise ValueError(f"Unknown weight init method: {self.weight_init_method}")

        self.weights = w

    def find_bmu(self, x):
        diff = self.weights - x[np.newaxis, np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        return np.unravel_index(np.argmin(dists), (self.m, self.n))

    # calculates the distance between neuron (i1, j1) and (i2, j2) in the grid-space
    def grid_distance(self, i1, j1, i2, j2):
        a = np.array([i1, j1])
        b = np.array([i2, j2])
        diff = np.abs(a - b)

        if self.grid_metric == "euclid":
            return np.linalg.norm(diff)
        elif self.grid_metric == "manhattan":
            return np.sum(diff)
        elif self.grid_metric == "chebyshev":
            return np.max(diff)
        elif self.grid_metric == "cosine":
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 and norm_b == 0:
                return 1.0
            if norm_a == 0 or norm_b == 0:
                # if we want neutral evaluation, we use 0.5
                return 0.5
            # standard calculation for cosine similarity
            return 1 - (np.dot(a, b) / (norm_a * norm_b + 1e-12))
        elif self.grid_metric == "toroidal":
            di = min(abs(i1 - i2), self.m - abs(i1 - i2))
            dj = min(abs(j1 - j2), self.n - abs(j1 - j2))
            return np.sqrt(di**2 + dj**2)
        else:
            raise ValueError(f"Unknown grid_metric: {self.grid_metric!r}")

    def compute_neighborhood(self, d, radius):
        if self.neighborhood_kernel == "gaussian":
            h = np.exp(-(d**2) / (2 * radius**2))
        elif self.neighborhood_kernel == "bubble":
            h = np.where(d <= radius, 1.0, 0.0)
        elif self.neighborhood_kernel == "epanechnikov":
            z = (d**2) / (radius**2)
            h = np.maximum(0, 1 - z)
        elif self.neighborhood_kernel == "triangular":
            h = np.maximum(0, 1 - (d / radius))
        elif self.neighborhood_kernel == "inverse":
            h = 1 / (1 + (d / radius)) if radius > 0 else 0
        else:
            raise ValueError(
                f"Unknown neighborhood kernel: {self.neighborhood_kernel!r}"
            )

        return h

    def update_weights(self, x, bmu, lr, radius):
        i0, j0 = bmu
        for i in range(self.m):
            for j in range(self.n):
                d = self.grid_distance(i, j, i0, j0)
                h = self.compute_neighborhood(d, radius)
                self.weights[i, j] += lr * h * (x - self.weights[i, j])

    def train(self, data, num_epochs: int = 100):
        data = np.asarray(data).reshape(-1, self.dim)
        self.init_weights(data)

        self.q_error = []
        self.avg_adjust = []

        for epoch in range(num_epochs):
            perm = np.random.permutation(len(data))
            data = data[perm]

            old_weights = self.weights.copy()
            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)

            for x in data:
                bmu = self.find_bmu(x)
                self.update_weights(x, bmu, lr, radius)

            # quantization error
            dists = [np.linalg.norm(x - self.weights[self.find_bmu(x)]) for x in data]
            self.q_error.append(np.mean(dists))

            # average adjustment
            delta = self.weights - old_weights
            self.avg_adjust.append(np.mean(np.linalg.norm(delta.reshape(-1, self.dim), axis=1)))
