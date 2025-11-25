import numpy as np

from .SOM import SOM

class RSOM(SOM):
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
        """
        Shapes: 
        self.weights:        (m, n, dim)
        self.context_weights: (m*n, m*n)
        self.context_vector:  (m*n,)
        """
        
        """
        Input space distance:
        Flatten the 2D grid of neurons into one list of m*n neurons
        Each neuron has a weight of length dim
        """
        W_flat = self.weights.reshape(self.m * self.n, self.dim) # (m*n, dim)
        # compute Euclidean distance between x and each neuron's input weight vector
        diff_input = W_flat - x[np.newaxis, :] # (m*n, dim)
        d_input = np.sum(diff_input**2, axis=1) # (m*n,)
        """
        Compute context distances:
        Each neuron i has a context weight vector C_i (row i of context weights)
        Compare each C_i with the current context vector c_t
        """
        diff_context = self.context_weights - self.context_vector[np.newaxis, :] # (m*n, m*n)
        d_context = np.sum(diff_context**2, axis=1) # (m*n,)
        # combine distances
        d_comb = self.alpha * d_input + self.beta * d_context
        # return the 2D position of neuron with minimal distance
        return np.unravel_index(np.argmin(d_comb), (self.m, self.n))

    def update_weights(self, x, bmu, lr, radius):
        """
        self.weights[i,j]: (dim,)
        self.context_weights: (m*n, m*n)
        self.context_weights[k]: (m*n,) -> context weight vector of neuron k
        self.context_vector: (m*n,)
        """
        i0, j0 = bmu # BMU coordinates
        ct = self.context_vector # current context vector (m*n,)
        for i in range(self.m):
            for j in range(self.n):
                # compute grid‐distance between (i,j) and the BMU
                d = self.grid_distance(i, j, i0, j0)
                h = self.compute_neighborhood(d, radius)
                # apply update
                self.weights[i, j] += lr * self.gamma1 * h * (x - self.weights[i, j])
                # context weight vector update
                index = i * self.n + j
                self.context_weights[index] += lr * self.gamma2 * h * (ct - self.context_weights[index])

    def update_context(self, x):
        # recompute context_vector using exponential activation on combined distances
        W_flat = self.weights.reshape(self.m * self.n, self.dim)
        diff_input = W_flat - x[np.newaxis, :]
        d_input = np.sum(diff_input ** 2, axis=1)
        C = self.context_weights
        ct_old = self.context_vector
        d_context = np.sum((C - ct_old[np.newaxis, :]) ** 2, axis=1)
        d_comb = self.alpha * d_input + self.beta * d_context
        # exponential activation
        """
        If exponential activation (exp(-d_comb)) is too sensitive you can stabilize the context state magnitudes
        ct_new = np.exp(-d_comb)
        self.context_vector = ct_new / np.sum(ct_new)
        """
        self.context_vector = np.exp(-d_comb)

    def train(self, data, num_epochs: int = 100):
        data = np.asarray(data).reshape(-1, self.dim)
        self.init_weights(data)

        Q = self.m * self.n
        self.context_weights = np.random.rand(Q, Q)
        self.context_vector = np.zeros(Q)

        self.q_error = []
        self.avg_adjust_main = []
        self.avg_adjust_context = []
        self.context_norms = []
        self.bmu_trajectory = []
        self.context_history = []

        for epoch in range(num_epochs):
            perm = np.random.permutation(len(data))
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
            
            dists = [np.linalg.norm(x - self.weights[self.find_bmu(x)]) for x in data]
            self.q_error.append(np.mean(dists))

            delta_main = np.abs(self.weights - old_main_weights)
            self.avg_adjust_main.append(np.mean(delta_main))
            delta_context = np.abs(self.context_weights - old_context_weights)
            self.avg_adjust_context.append(np.mean(delta_context))

            self.context_norms.append(np.linalg.norm(self.context_vector))
