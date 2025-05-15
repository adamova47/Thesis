import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Union


class SOM:
    def __init__(self,
                 m: int, # number of rows of the map
                 n: int, # number of columns of the map
                 dim: int, # dimensionality of input data
                 *,
                 weight_init_method: Union[str, callable] = 'uniform', # string naming a built-in method or callable function
                 grid_metric: str = 'euclid',
                 neighborhood_kernel: str = 'gaussian',  # 'gaussian','bubble','mexican_hat',...
                 lr_schedule : callable = None,  # fn(epoch,max_epochs) → lr
                 radius_schedule: callable = None,  # fn(epoch,max_epochs) → radius
                 seed = None
                 ):

        # set a random seed to get a consistent result from the random number generator
        if seed is not None:
            np.random.seed(seed)

        # stores the grid dimensions and input dimensionality - they must be positive or reshape() will fail later on
        self.m, self.n = m, n
        self.dim = dim
        # remembers the method we want to use to initialize weights
        # could be 'uniform', 'data_range', 'sample', 'pca', 'kmeans' or a custom user function
        self.weight_init_method = weight_init_method
        self.weights = None

        # distance on the 2D grid: 'l1','l2','chebyshev','minkowski','toroidal'
        self.grid_metric = grid_metric
        # neighborhood shape: 'gaussian', 'bubble', 'mexican_hat', 'epanechnikov', 'triangular', 'inverse'
        self.neighborhood_kernel = neighborhood_kernel

        # default schedules: linear decay
        self.lr_schedule = lr_schedule or (lambda t, T: 0.1 * (1 - t / T))
        self.radius_schedule = radius_schedule or (lambda t, T: max(m, n) / 2 * (1 - t / T))

        # metrics history
        self.q_error_history = []
        self.avg_adjust_history = []


    def init_weights(self, data):
        # if a user passes a function, we just call it
        # expect signature: fn(data, m, n, dim) → array of shape (m, n, dim)
        if callable(self.weight_init_method):
            w = self.weight_init_method(data, self.m, self.n, self.dim)

        elif self.weight_init_method == 'uniform':
            # purely random distribution in the range [0, 1), data isn't needed
            w = np.random.rand(self.m, self.n, self.dim)

        elif self.weight_init_method == 'data_range':
            # for each feature k: sample uniformly from [min_k, max_k]
            lo = data.min(axis=0) # shape (dim,)
            hi = data.max(axis=0) # shape (dim,)
            # broadcast span (dim,) across (m, n, dim)
            w = (hi - lo) * np.random.rand(self.m, self.n, self.dim) + lo

        elif self.weight_init_method == 'sample':
            # randomly picks m*n rows (with replacement) from the data
            # if the data has less than m*n rows, replace = True avoids error
            idx = np.random.choice(data.shape[0], self.m * self.n, replace=True)
            # reshape back to grid
            w = data[idx].reshape(self.m, self.n, self.dim)

        elif self.weight_init_method == 'pca':
            """
            # `pca.components_` is a (2 × dim) matrix whose rows are the principal axes.
            # `pca.explained_variance_` gives the variance along each axis.
            # `pca.mean_` is the data mean in R^dim.
            """
            # PCA on the data -> 2 principal axes
            # fails if the data.ndim < 2 or if variance in any PC is zero
            pca = PCA(n_components=2).fit(data)
            # build coordinates of regular (m * n) grin in [0,1]^2
            coords = np.indices((self.m, self.n)).transpose(1, 2, 0).astype(float)
            coords[..., 0] /= (self.m - 1)
            coords[..., 1] /= (self.n - 1)
            # scale grid by sqrt(eigenvalues) to match data variance
            pc_grid = coords.reshape(-1, 2) * np.sqrt(pca.explained_variance_)
            # project back into the original d-space and add the mean
            w = pc_grid @ pca.components_ # shape (m*n, dim)
            w += pca.mean_ # broadcast add
            w = w.reshape(self.m, self.n, self.dim)
        elif self.weight_init_method == 'kmeans':
            # KMeans to find m*n centroids
            # can be slow on large data / large m*n
            km = KMeans(n_clusters=self.m * self.n, n_init=10).fit(data)
            w = km.cluster_centers_.reshape(self.m, self.n, self.dim)

        else:
            raise ValueError(f"Unknown weight init method: {self.weight_init_method}")

        self.weights = w

    def find_bmu(self, x):
        # self.weights has shape (m, n, dim)
        # subtract x from every weight -> (m, n, dim)
        diff = self.weights - x[np.newaxis, np.newaxis, :]
        # compute Euclidean distance along the last axis -> (m, n)
        dists = np.linalg.norm(diff, axis=2)
        # unravel the flattened argmin back into 2D coordinates
        return np.unravel_index(np.argmin(dists), (self.m, self.n))

    # calculates the distance between neuron (i1, j1) and (i2, j2) in the grid-space
    def grid_distance(self, i1, j1, i2, j2):
        a = np.array([i1, j1])
        b = np.array([i2, j2])
        diff = np.abs(a - b)

        if self.grid_metric == 'euclid':
            return np.linalg.norm(diff)
        elif self.grid_metric == 'manhattan':
            return np.sum(diff)
        elif self.grid_metric == 'chebyshev':
            return np.max(diff)
        elif self.grid_metric == 'cosine':
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 and norm_b == 0:
                return 1.0
            if norm_a == 0 or norm_b == 0:
                # if we want neutral evaluation, we use 0.5
                return 0.5
            # standard calculation for cosine similarity
            return 1 - (np.dot(a, b) / (norm_a * norm_b))
        elif self.grid_metric == 'toroidal':
            di = min(abs(i1 - i2), self.m - abs(i1 - i2))
            dj = min(abs(j1 - j2), self.n - abs(j1 - j2))
            return np.sqrt(di ** 2 + dj ** 2)
        else:
            raise ValueError(f"Unknown grid_metric: {self.grid_metric!r}")


    def compute_neighborhood(self, d, radius):
        # compute the neighborhood kernel h(d, radius)
        if self.neighborhood_kernel == 'gaussian':
            """
            smooth decay: neurons farther from the BMU get exponentially less update
            pros: very common; yields smooth, continuous maps; never exactly zero so truly “continuous” update
            cons: requires computing an exponential; theoretically infinite support (though numerically tiny far away)
            """
            h = np.exp(-(d ** 2) / (2 * radius ** 2))
        elif self.neighborhood_kernel == 'bubble':
            """
            all or nothing: neurons inside the radius get an update, everything else gets ignored
            pros: very simple and fast to compute
            cons: can produce blocky maps, with sharp boundaries between regions
            """
            h = 1 if d <= radius else 0
        elif self.neighborhood_kernel == 'epanechnikov':
            """
            finite support: zero beyond d > r, quadratic decay inside
            pros: simpler than gaussian, strictly local
            cons: less smooth at the cutoff 
            """
            z = (d ** 2) / (radius ** 2)
            h = np.maximum(0, 1 - z)
        elif self.neighborhood_kernel == 'triangular':
            """
            linear decay inside the radues, zero outside
            """
            h = np.maximum(0, 1 - (d / radius))
        elif self.neighborhood_kernel == 'inverse':
            """
            heavy tail:never zero but decaysonly 1/d
            """
            h = 1.0 / (1.0 + (d / radius))
        else:
            raise ValueError(f"Unknown neighborhood kernel: {self.neighborhood_kernel!r}")

        return h

    def update_weights(self, x, bmu, lr, radius):
        i0, j0 = bmu
        for i in range(self.m):
            for j in range(self.n):
                # compute grid‐distance between (i,j) and the BMU
                d = self.grid_distance(i, j, i0, j0)
                h = self.compute_neighborhood(d, radius)
                # Apply update
                self.weights[i, j] += lr * h * (x - self.weights[i, j])

    def train(self, data, num_epochs: int = 100, mode: str = 'online'):
        data = np.asarray(data).reshape(-1, self.dim)
        self.q_error_history = []
        self.avg_adjust_history = []

        self.init_weights(data)

        for epoch in range(num_epochs):
            # shuffle the order of data once per epoch to avoid bias in maps early updates and slow convergence
            perm = np.random.permutation(len(data))
            data = data[perm]

            old_weights = self.weights.copy()
            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)

            if mode == 'online':
                for x in data:
                    bmu = self.find_bmu(x)
                    self.update_weights(x, bmu, lr, radius)

                # quantization error
                dists = [np.linalg.norm(x - self.weights[self.find_bmu(x)]) for x in data]
                self.q_error_history.append(np.mean(dists))

                # average adjustment
                delta = self.weights - old_weights
                self.avg_adjust_history.append(np.mean(np.linalg.norm(delta.reshape(-1, self.dim), axis=1)))

            elif mode == 'batch':
                ...
                # TODO
            else:
                raise ValueError(f"Unknown mode: {mode!r}")