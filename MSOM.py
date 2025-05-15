import numpy as np

from SOM import SOM

class MSOM(SOM):
    """
    Merge Self-Organizing Map (MSOM):
    - each neuron has:
      * weight w^i in R^dim
      * context weight c^i in R^dim
    - context descriptor c^t = (1 - beta) * w^{I_{t-1}} + beta * c^{I_{t-1}}
    - distance: d_i(t) = (1 - alpha) * ||x^t - w^i||^2 + alpha * ||c^t - c^i||^2
    - weight updates:
        △w^i = gamma1 * h(d_N(i,I_t)) * (x^t - w^i)
        △c^i = gamma2 * h(d_N(i,I_t)) * (c^t - c^i)
    """
    def __init__(self,
                 m: int,
                 n: int,
                 dim: int,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma1: float = 0.1,
                 gamma2: float = 0.1,
                 *,
                 weight_init_method='uniform',
                 grid_metric='euclid',
                 neighborhood_kernel='gaussian',
                 lr_schedule=None,
                 radius_schedule=None,
                 seed=None
                 ):
        super().__init__(
            m=m,
            n=n,
            dim=dim,
            weight_init_method=weight_init_method,
            grid_metric=grid_metric,
            neighborhood_kernel=neighborhood_kernel,
            lr_schedule=lr_schedule,
            radius_schedule=radius_schedule,
            seed=seed
        )
        self.alpha = alpha  # distance mixing between input and context
        self.beta = beta  # context merge rate
        self.gamma1 = gamma1  # learning rate for weights
        self.gamma2 = gamma2  # learning rate for context weights

        self.context_weights = None
        self.context_vector = None

    def find_bmu(self, x):
        # compute distances on grid
        # d_input[i,j] = ||x - W[i,j]||^2
        diff_w = self.weights - x[np.newaxis, np.newaxis, :]
        d_input = np.sum(diff_w ** 2, axis=2)
        # d_context[i,j] = ||c_t - C[i,j]||^2
        diff_c = self.context_weights - self.context_vector[np.newaxis, np.newaxis, :]
        d_context = np.sum(diff_c ** 2, axis=2)
        # combined distance
        d_comb = (1 - self.alpha) * d_input + self.alpha * d_context
        # best unit
        return np.unravel_index(np.argmin(d_comb), (self.m, self.n))

    def update_weights(self, x, bmu, lr, radius):
        i0, j0 = bmu
        ct = self.context_vector
        for i in range(self.m):
            for j in range(self.n):
                d = self.grid_distance(i, j, i0, j0)
                h = self.compute_neighborhood(d, radius)
                # update weight vector
                self.weights[i, j] += self.gamma1 * lr * h * (x - self.weights[i, j])
                # update context weight vector
                self.context_weights[i, j] += self.gamma2 * lr * h * (ct - self.context_weights[i, j])

    def update_context(self, bmu):
        # merge context from winner
        w_win = self.weights[bmu]
        c_win = self.context_weights[bmu]
        self.context_vector = (1 - self.beta) * w_win + self.beta * c_win

    def train(self, data, num_epochs: int = 100, mode: str = 'online'):
        data = np.asarray(data).reshape(-1, self.dim)
        # initialize weights
        self.init_weights(data)
        # initialize context weights
        self.context_weights = np.random.rand(self.m, self.n, self.dim)
        # initial context descriptor
        self.context_vector = np.zeros(self.dim)

        for epoch in range(num_epochs):
            perm = np.random.permutation(len(data))
            data_shuffled = data[perm]
            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)

            if mode == 'online':
                for x in data_shuffled:
                    bmu = self.find_bmu(x)
                    self.update_weights(x, bmu, lr, radius)
                    self.update_context(bmu)

            elif mode == 'batch':
                pass
            else:
                raise ValueError(f"Unknown mode: {mode!r}")
