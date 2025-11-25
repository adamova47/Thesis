import cupy as cp

DEFAULT_DTYPE = cp.float32

class SOM_vectorized:
    def __init__(self, m: int, n: int, dim: int, 
                 weight_init_method: str = "uniform",
                 grid_metric: str = "euclid",
                 neighborhood_kernel: str = "gaussian",
                 lr_schedule: callable = None,
                 radius_schedule: callable = None,
                 seed=None):

        if seed is not None:
            cp.random.seed(seed)

        if m <= 0 or n <= 0 or dim <= 0:
            raise ValueError("SOM dimensions must be positive integers")
        
        self.m = m  # number of rows of the map
        self.n = n  # number of columns of the map
        self.dim = dim  # dimensionality of input data

        self.weight_init_method = weight_init_method
        self.weights = None
        self.grid_metric = grid_metric
        self.neighborhood_kernel = neighborhood_kernel

        self.lr_schedule = lr_schedule or (lambda t, T: 0.1 * (1 - t / T))
        self.radius_schedule = radius_schedule or (lambda t, T: max(m, n) / 2 * (1 - t / T))

        self.q_error_history = []
        self.avg_adjust_history = []

        self._precompute_grid_coords()


    def init_weights(self, data):
        if self.weight_init_method == "uniform":
            self.weights = cp.random.rand(self.m, self.n, self.dim)
        elif self.weight_init_method == "data_range":
            lo = data.min(axis=0)
            hi = data.max(axis=0)
            w = (hi - lo) * cp.random.rand(self.m, self.n, self.dim) + lo
        elif self.weight_init_method == "sample":
            indices = cp.random.choice(data.shape[0], size=self.m * self.n, replace=True)
            self.weights = data[indices].reshape(self.m, self.n, self.dim)
        elif self.weight_init_method == "pca":
            from sklearn.decomposition import PCA
            data_np = cp.asnumpy(data)
            pca = PCA(n_components=2).fit(data_np)
            coords = cp.indices((self.m, self.n)).transpose(1, 2, 0).astype(float)
            coords[..., 0] /= max(1, self.m - 1)
            coords[..., 1] /= max(1, self.n - 1)
            if self.m > 1 and self.n > 1:
                coords[..., 0] = coords[..., 0] * (pca.explained_variance_[0] * 2) - pca.explained_variance_[0]
                coords[..., 1] = coords[..., 1] * (pca.explained_variance_[1] * 2) - pca.explained_variance_[1]
            pc_grid = coords.reshape(-1, 2)
            weights = pc_grid @ pca.components_
            weights += pca.mean_
            weights = weights.reshape(self.m, self.n, self.dim)
            self.weights = cp.asarray(weights, dtype=DEFAULT_DTYPE)
        elif self.weight_init_method == "kmeans":
            from sklearn.cluster import KMeans
            data_np = cp.asnumpy(data)
            km = KMeans(n_clusters=self.m * self.n, n_init=10).fit(data_np)
            centers = km.cluster_centers_.reshape(self.m, self.n, self.dim)
            self.weights = cp.asarray(centers, dtype=DEFAULT_DTYPE)

    def _precompute_grid_coords(self):
        """Precompute grid coordinates for efficient distance calculation"""
        self.i_coords, self.j_coords = cp.mgrid[0:self.m, 0:self.n]

    def find_bmu(self, x):
        """Most efficient BMU finding using einsum or manual computation"""
        # Method 1: Using einsum (very efficient, works with CuPy)
        diff = self.weights - x  # Broadcasting: (m, n, dim) - (dim,) → (m, n, dim)
        dists_sq = cp.einsum('ijk,ijk->ij', diff, diff)  # Sum of squares
        min_index = cp.argmin(dists_sq)  # argmin of squared distances
        return (min_index // self.n, min_index % self.n)
        
        # Method 2: Manual computation (also good)
        # diff = self.weights - x
        # dists_sq = cp.sum(diff * diff, axis=2)
        # min_index = cp.argmin(dists_sq)
        # return (min_index // self.n, min_index % self.n)

    def grid_distance(self, i0, j0):
        if self.grid_metric == "euclid":
            return cp.sqrt((self.i_coords - i0)**2 + (self.j_coords - j0)**2)
        elif self.grid_metric == "manhattan":
            return cp.abs(self.i_coords - i0) + cp.abs(self.j_coords - j0)
        elif self.grid_metric == "chebyshev":
            return cp.maximum(cp.abs(self.i_coords - i0), cp.abs(self.j_coords - j0))
        elif self.grid_metric == "toroidal":
            di = cp.minimum(cp.abs(self.i_coords - i0), self.m - cp.abs(self.i_coords - i0))
            dj = cp.minimum(cp.abs(self.j_coords - j0), self.n - cp.abs(self.j_coords - j0))
            return cp.sqrt(di**2 + dj**2)

    def compute_neighborhood(self, dists, radius):
        if self.neighborhood_kernel  == "gaussian":
            return cp.exp(-(dists**2) / (2 * radius**2))
        elif self.neighborhood_kernel  == "bubble":
            return (dists <= radius).astype(cp.float32)
        elif self.neighborhood_kernel  == "epanechnikov":
            z = (dists**2) / (radius**2)
            return cp.maximum(0, 1 - z)
        elif self.neighborhood_kernel  == "triangular":
            return cp.maximum(0, 1 - (dists / radius))
        elif self.neighborhood_kernel  == "inverse":
            return 1.0 / (1.0 + (dists / radius))

    def update_weights(self, x, bmu, lr, radius):
        i0, j0 = bmu

        dists = self.grid_distance(i0, j0)
        h = self.compute_neighborhood(dists, radius)

        # Apply update to ALL neurons at once
        update = lr * h[:, :, cp.newaxis].astype(self.weights.dtype) * (x - self.weights)
        self.weights += update


    def train(self, data, num_epochs = 100):
        training_data = cp.asarray(data, dtype=DEFAULT_DTYPE)
        training_data = training_data.reshape(-1, self.dim)
        self.init_weights(training_data)

        self.q_error_history = []
        self.avg_adjust_history = []

        for epoch in range(num_epochs):
            perm = cp.random.permutation(len(training_data))
            data_epoch = training_data[perm]

            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)
            old_weights = self.weights.copy()

            total_q_error = 0.0
            for x in data_epoch:
                bmu = self.find_bmu(x)
                self.update_weights(x, bmu, lr, radius)
                total_q_error += cp.linalg.norm(x - self.weights[bmu])

            self.q_error_history.append(total_q_error / len(data_epoch))
            delta = self.weights - old_weights
            avg_adjust = cp.mean(cp.linalg.norm(delta.reshape(-1, self.dim), axis=1))
            self.avg_adjust_history.append(avg_adjust)