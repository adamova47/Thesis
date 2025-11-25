import cupy as cp
from .SOM_vectorized import SOM_vectorized
DEFAULT_DTYPE = cp.float32

class RSOM(SOM_vectorized):
    """
    RecSOM: Recurrent Self-Organizing Map
    - two layers:
      * weights W (shape m×n×dim)
      * context weights (shape (m*n)×(m*n))
    - context vector ct (shape m*n), representing context layer output
    - distance mix: d = alpha * ||x - W_ij||^2 + beta * ||ct - C_ij||^2
    - updateWeights:
        W_ij += lr * gamma1 * h * (x - W_ij)
        C_ij_vec += lr * gamma2 * h * (ct - C_ij_vec)
    - update context after each input x:
        for each neuron k: compute d_k = alpha||x-W_k||^2 + beta||ct - C_k||^2
        ct = exp(-d)  (exponential activation)
    """

    def __init__(self,
                 m: int,
                 n: int,
                 dim: int,
                 alpha: float = 0.1,
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
        
        super().__init__(m=m, n=n, dim=dim,
                         weight_init_method=weight_init_method,
                         grid_metric=grid_metric,
                         neighborhood_kernel=neighborhood_kernel,
                         lr_schedule=lr_schedule,
                         radius_schedule=radius_schedule,
                         seed=seed)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.context_weights = None
        self.context_vector = None

        self.avg_adjust_main = []
        self.avg_adjust_context = []
        self.context_norms = []
        self.bmu_trajectory = []
        self.context_history = []

    def find_bmu(self, x):
        W_flat = self.weights.reshape(self.m * self.n, self.dim)
        diff_input = W_flat - x[cp.newaxis, :]
        d_input = cp.sum(diff_input**2, axis=1)

        diff_context = self.context_weights - self.context_vector[cp.newaxis, :]
        d_context = cp.sum(diff_context**2, axis=1)

        d_comb = self.alpha * d_input + self.beta * d_context
        bmu_index = cp.argmin(d_comb)
        return cp.unravel_index(bmu_index, (self.m, self.n))

    def update_weights(self, x, bmu, lr, radius):
        i0, j0 = bmu
        ct = self.context_vector

        # Compute neighborhood for all neurons at once
        dists = self._grid_distance(i0, j0) # shape (m, n)
        h = self._neighborhood_function(dists, radius) # shape (m, n)
        h_expanded = h[:, :, None].astype(self.weights.dtype)  # (m,n,1) for broadcasting

        # Update main weights
        self.weights += lr * self.gamma1 * h_expanded * (x - self.weights)

        # Update context weights
        Q = self.m * self.n
        h_flat = h.reshape(Q).astype(self.context_weights.dtype)  # (Q,)
        self.context_weights += lr * self.gamma2 * h_flat[:, None] * (ct - self.context_weights)


    def update_context(self, x):
        Q = self.m * self.n
        # Flatten input weights
        W_flat = self.weights.reshape(Q, self.dim)  # (Q, dim)
        # Input distance
        diff_input = W_flat - x[None, :]  # (Q, dim)
        d_input = cp.sum(diff_input ** 2, axis=1)  # (Q,)
        # Context distance
        diff_context = self.context_weights - self.context_vector[None, :]  # (Q, Q)
        d_context = cp.sum(diff_context ** 2, axis=1)  # (Q,)
        # Combine distances
        d_comb = self.alpha * d_input + self.beta * d_context
        # Exponential activation
        self.context_vector = cp.exp(-d_comb)
        # exponential overflow might become tiny with large d_comb so we can also consider:
        # self.context_vector = cp.exp(-d_comb - cp.max(d_comb))

    def train(self, data, num_epochs: int = 100):
        data = cp.asarray(data).reshape(-1, self.dim)
        self.init_weights(data)

        Q = self.m * self.n
        self.context_weights = cp.random.rand(Q, Q, dtype=DEFAULT_DTYPE)
        self.context_vector = cp.zeros(Q, dtype=DEFAULT_DTYPE)

        self.q_error = []
        self.avg_adjust_main = []
        self.avg_adjust_context = []
        self.context_norms = []
        self.bmu_trajectory = []
        self.context_history = []

        for epoch in range(num_epochs):
            perm = cp.random.permutation(len(data))
            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)

            old_main_weights = self.weights.copy()
            old_context_weights = self.context_weights.copy()

            for x in data[perm]:
                bmu = self.find_bmu(x)
                self.update_weights(x, bmu, lr, radius)
                self.update_context(x)
                self.context_history.append(self.context_vector.copy())
                self.bmu_trajectory.append(bmu)
            
            dists = [cp.linalg.norm(x - self.weights[self.find_bmu(x)]) for x in data]
            self.q_error.append(cp.mean(dists))

            delta_main = cp.abs(self.weights - old_main_weights)
            self.avg_adjust_main.append(cp.mean(delta_main))
            delta_context = cp.abs(self.context_weights - old_context_weights)
            self.avg_adjust_context.append(cp.mean(delta_context))

            self.context_norms.append(cp.linalg.norm(self.context_vector))
