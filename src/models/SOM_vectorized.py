import cupy as cp


"""
CuPy gives NumPy-like arrays but on the GPU.
float32 is usually the sweet spot for the GPU performance.
If the data is float64 or comes from NumPy, CuPy will convert it, but
mixing types can hurt performance and cause slight precision differences.
"""
DEFAULT_DTYPE = cp.float32

class SOM_vectorized:

    def __init__(self, m: int, n: int, dim: int, 
                 weight_init_method: str = "uniform",
                 grid_metric: str = "euclid",
                 neighborhood_kernel: str = "gaussian",
                 lr0=0.1,
                 lr_final=0.001,
                 radius0=None,
                 radius_final=1.0,
                 lr_schedule: callable = None,
                 radius_schedule: callable = None,
                 seed=None):

        # Seeding ensures reproducibility within CuPy.
        if seed is not None:
            cp.random.seed(seed)

        # Basic sanity guard for dimensions.
        if m <= 0 or n <= 0 or dim <= 0:
            raise ValueError("SOM dimensions must be positive integers")
        
        self.m = m  # number of rows of the map
        self.n = n  # number of columns of the map
        self.dim = dim  # dimensionality of input data

        self.weight_init_method = weight_init_method
        self.weights = None
        self.grid_metric = grid_metric
        self.neighborhood_kernel = neighborhood_kernel

        # learning rate and radius lineary decay by default, but can be customized with a function of (current_epoch, total_epochs)
        self.lr0 = lr0
        self.lr_final = lr_final
        self.radius0 = radius0 if radius0 is not None else max(m, n) / 2
        self.radius_final = radius_final

        self.lr_schedule = lr_schedule or self._default_lr_schedule
        self.radius_schedule = radius_schedule or self._default_radius_schedule

        # we track quantization error and average adjustment for monitoring convergence
        self.q_error_history = []
        self.avg_adjust_history = []

        # precompute grid coordinates for efficient distance calculations during training
        # you avoid recomputing grid indices constantly
        self._precompute_grid_coords()


    def _default_lr_schedule(self, t, T):
        return self.lr0 * (self.lr_final / self.lr0) ** (t / T)


    def _default_radius_schedule(self, t, T):
        return self.radius0 * (self.radius_final / self.radius0) ** (t / T)


    def init_weights(self, data):
        """
        This controls what the SOM starts as, which can affect convergence speed and final quality.
        """
        if self.weight_init_method == "uniform":
            """
            This samples in [0, 1].
            The assumption is that the data is normalized or roughly in [0,1].
            If not the  SOM eventually learns but it can take longer or behave weirdly early on.
            """
            self.weights = cp.random.rand(self.m, self.n, self.dim).astype(DEFAULT_DTYPE)
        elif self.weight_init_method == "data_range":
            """
            This samples uniformly within the bounding box of the data.
            Which ensures the initial weights are in a reasonable range relative to the data.
            """
            lo = data.min(axis=0)
            hi = data.max(axis=0)
            w = (hi - lo) * cp.random.rand(self.m, self.n, self.dim) + lo
            self.weights = w.astype(DEFAULT_DTYPE)
        elif self.weight_init_method == "sample":
            """
            Initialize each neuron as a random data point.
            Its good because the weights start at the  data manifold and often converge faster than uniform for example.
            The Replace = True might reduce diversity though, but it ensures it works even if you have fewer data points than neurons.
            """
            indices = cp.random.choice(data.shape[0], size=self.m * self.n, replace=True)
            self.weights = data[indices].reshape(self.m, self.n, self.dim)
        elif self.weight_init_method == "pca":
            """
            You lay your grid on the first 2 principal components of the data.
            It often makes training smoother and faster but only makes sense if dim is not tiny and the PCA structure matters.
            """
            if self.dim < 2 or data.shape[0] < 2:
                # fallback for 1D or too-few-samples cases
                indices = cp.random.choice(data.shape[0], size=self.m * self.n, replace=True)
                self.weights = data[indices].reshape(self.m, self.n, self.dim).astype(DEFAULT_DTYPE)
                return
            from sklearn.decomposition import PCA
            data_np = cp.asnumpy(data)
            pca = PCA(n_components=2).fit(data_np)
            pca_components = cp.asarray(pca.components_, dtype=DEFAULT_DTYPE)
            pca_mean = cp.asarray(pca.mean_, dtype=DEFAULT_DTYPE)
            pca_var = cp.asarray(pca.explained_variance_, dtype=DEFAULT_DTYPE)
            coords = cp.indices((self.m, self.n)).transpose(1, 2, 0).astype(DEFAULT_DTYPE)
            coords[..., 0] /= max(1, self.m - 1)
            coords[..., 1] /= max(1, self.n - 1)
            if self.m > 1 and self.n > 1:
                coords[..., 0] = coords[..., 0] * (2 * pca_var[0]) - pca_var[0]
                coords[..., 1] = coords[..., 1] * (2 * pca_var[1]) - pca_var[1]
            pc_grid = coords.reshape(-1, 2)
            weights = (pc_grid @ pca_components) + pca_mean
            self.weights = weights.reshape(self.m, self.n, self.dim)
        elif self.weight_init_method == "kmeans":
            from sklearn.cluster import KMeans
            data_np = cp.asnumpy(data)
            km = KMeans(n_clusters=self.m * self.n, n_init=10).fit(data_np)
            centers = km.cluster_centers_.reshape(self.m, self.n, self.dim)
            self.weights = cp.asarray(centers, dtype=DEFAULT_DTYPE)

    def _precompute_grid_coords(self):
        """
        This creates 2 arrays of shape (m, n) that contain the i and j indices of the grid.
            - self.i_coords[i, j] = i
            - self.j_coords[i, j] = j
        Which allows us to compute distances from the BMU to all neurons in a vectorized way during training, without needing to recompute the grid indices each time.
        """
        self.i_coords, self.j_coords = cp.mgrid[0:self.m, 0:self.n]

    def find_bmu(self, x):
        """
        Most efficient BMU finding using einsum or manual computation.
        For each neuron, compute squared Euclidean distance then pick the neuron with the smallest distance.
        """
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
        """
        Distance between neurons in grid coordinate space.
            - For Euclidean, it's the straight line distance.
            - For Manhattan, it's the sum of absolute differences.
            - For Chebyshev, it's the maximum absolute difference.
            - For Toroidal, it wraps around the edges of the grid.
        We compute the distance from the BMU (i0, j0) to all neurons in the grid at once using the precomputed coordinates.
        """
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
        radius = max(float(radius), 1e-8)
        # How influence decays with grid distance.
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


    def train(self, 
              data, 
              num_epochs = 100,
              min_epochs: int = 75,
              patience: int = 8,
              min_delta: float = 1e-4
              ):
        training_data = cp.asarray(data, dtype=DEFAULT_DTYPE)
        training_data = training_data.reshape(-1, self.dim)
        self.init_weights(training_data)

        self.q_error_history = []
        self.avg_adjust_history = []

        best_metric = float("inf")
        best_weights = None
        wait = 0
        best_epoch = -1

        for epoch in range(num_epochs):
            perm = cp.random.permutation(len(training_data))
            data_epoch = training_data[perm]

            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)
            old_weights = self.weights.copy()

            total_q_error = 0.0
            for x in data_epoch:
                bmu = self.find_bmu(x)
                w_bmu = self.weights[bmu].copy()
                total_q_error += cp.linalg.norm(x - w_bmu)
                self.update_weights(x, bmu, lr, radius)

            q_error = float((total_q_error / len(data_epoch)).item())
            self.q_error_history.append(q_error)

            delta = self.weights - old_weights
            avg_adjust = cp.mean(cp.linalg.norm(delta.reshape(-1, self.dim), axis=1))
            self.avg_adjust_history.append(float(avg_adjust.item()))

            if (best_metric - q_error) > min_delta:
                best_metric = q_error
                best_weights = self.weights.copy()
                best_epoch = epoch
                wait = 0
            elif epoch + 1 >= min_epochs:
                wait += 1
                if wait >= patience:
                    break
        
        if best_weights is not None:
            self.weights = best_weights

        self.best_q_error = best_metric
        self.best_epoch = best_epoch