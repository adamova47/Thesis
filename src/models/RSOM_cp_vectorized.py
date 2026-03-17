import cupy as cp
from .SOM_vectorized import SOM_vectorized
DEFAULT_DTYPE = cp.float32

class RSOM(SOM_vectorized):
    def __init__(self,
                 m: int,
                 n: int,
                 dim: int,
                 alpha: float = 0.1, # controls how much the input matters relative to memory (alpha = 1 behaves like a SOM and alpha = 0 behaves like a pure memory matcher)
                 beta: float = 0.5, # scaling factor for recurrent (context) distance term
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

        """
        Each neuron has its own expected context.
        Its analogous to SOM weights, but for history instead of input.
        """
        self.context_weights = None
        """
        Global context: previous activity vector y(t-1).
        Shape: (Q,), where Q = m*n.
        """
        self.context_vector = None

        self.avg_adjust_main = []
        self.avg_adjust_context = []
        self.context_norms = []
        self.bmu_trajectory = []
        self.context_history = []
        self.temporal_q_error_history = []

        self.best_epoch = -1

    def _compute_energy_and_activity(self, x):
        """
        Compute:
            d_i(t) = alpha * ||x - w_i||^2
                + beta  * ||y(t-1) - c_i||^2
        Then:
            y_i(t) = exp(-d_i(t))
        Returns:
            d_i   : (Q,) energy values
            y     : (Q,) activity vector
            bmu   : (i, j) grid coordinates
        """
        Q = self.m * self.n
        # Flatten input weights
        W_flat = self.weights.reshape(Q, self.dim)  # (Q, dim)
        # Input distance
        diff_input = W_flat - x[None, :]  # (Q, dim)
        d_input = cp.sum(diff_input ** 2, axis=1)  # (Q,)
        # Context distance
        diff_context = self.context_weights - self.context_vector[None, :]  # (Q, Q)
        d_context = cp.sum(diff_context ** 2, axis=1)  # (Q,)
        """y = self.context_vector
        y_norm = cp.sum(y.astype(cp.float64) ** 2)
        c_norm = cp.sum(self.context_weights.astype(cp.float64) ** 2, axis=1)
        dot = (self.context_weights.astype(cp.float64) @ y.astype(cp.float64))
        d_context = (c_norm + y_norm - 2 * dot).astype(cp.float32)"""
        # Combined energy
        d_comb = self.alpha * d_input + self.beta * d_context  # (Q,)
        # Activity vector (Voegtlin)
        """
        You can do: y = cp.exp(-d_i - cp.max(d_i))
        just max subtraction for numerical stability, but it also changes the scale of the context vector 
        which might affect learning dynamics. You can experiment with both versions.
        """
        y = cp.exp(-d_comb)  # (Q,)
        # Find BMU
        bmu_index = cp.argmin(d_comb)
        bmu = cp.unravel_index(bmu_index, (self.m, self.n))
        return d_comb, y, bmu

    def update_weights(self, x, bmu, lr, radius):
        """
        Voegtlin RecSOM updates:
            w_i^x <- w_i^x + lr * h_ib * (x(t)     - w_i^x)
            w_i^y <- w_i^y + lr * h_ib * (y(t-1)   - w_i^y)
        """
        i0, j0 = bmu
        y_prev = self.context_vector  # (Q,)
        # Compute neighborhood function
        dists = self.grid_distance(i0, j0)  # (m, n)
        h = self.compute_neighborhood(dists, radius)  # (m, n)
        h = h.astype(self.weights.dtype)
        # Update input weights (m, n, dim)
        self.weights += lr * h[:, :, None] * (x - self.weights)  # (m, n, dim)
        # Update recurrent/context weights (Q, Q)
        Q = self.m * self.n
        h_flat = h.reshape(Q).astype(self.context_weights.dtype)  # (Q,)
        # Each row i of context_weights gets scaled by h_flat[i] and multiplied by y_prev
        self.context_weights += lr * (h_flat[:, None] * (y_prev - self.context_weights))  # (Q, Q)

    def train(self, 
              data, 
              num_epochs: int = 100,
              min_epochs: int = 100,
              patience: int = 5,
              min_delta: float = 1e-4
              ):
        data = data.reshape(-1, self.dim)
        self.init_weights(data)

        Q = self.m * self.n
        dtype = self.weights.dtype
        """
        You can also initialize the recurrent weights in a more stable manner. Rand in [0, 1] is okay, but many implementations
        use small values around 0 so you can do:
        self.context_weights = 0.01 * cp.random.randn(Q, Q, dtype=dtype)
        """
        self.context_weights = cp.random.rand(Q, Q, dtype=dtype)
        self.context_vector = 0.01 * cp.zeros(Q, dtype=dtype)

        self.q_error_history = []
        self.temporal_q_error_history = []
        self.avg_adjust_main = []
        self.avg_adjust_context = []
        # self.context_norms = []
        self.bmu_trajectory = []
        # self.context_history = []

        # early stopping / checkpointing
        best_temporal_qe = float("inf")
        self.best_epoch = -1
        wait = 0

        best_weights = None
        best_context_weights = None
        best_context_vector = None

        for epoch in range(num_epochs):
            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)

            # old_main_weights = self.weights.copy()
            # old_context_weights = self.context_weights.copy()

            for x in data:
                d_comb, y, bmu = self._compute_energy_and_activity(x)
                self.update_weights(x, bmu, lr, radius)
                self.context_vector = y  # Update global context to current activity
            
            # Static + temporal QE computation
            old_context_vector = self.context_vector.copy()
            """
            We copy old context here to still have persistent sequence state across epochs. If its ok to reset context 
            each epoch you can just do self.context_vector = cp.zeros(Q, dtype=dtype) and not save the previous.
            """
            self.context_vector = cp.zeros(Q, dtype=dtype)

            static_err = 0.0
            temporal_err = 0.0

            for x in data:
                d_comb, y, bmu = self._compute_energy_and_activity(x)

                # input-only error
                w_bmu = self.weights[bmu]
                static_err += cp.linalg.norm(x - w_bmu)

                # temporal error
                bmu_idx = bmu[0] * self.n + bmu[1]
                temporal_err += cp.sqrt(d_comb[bmu_idx])

                # advance recurrent state during evaluation
                self.context_vector = y

            static_qe = static_err / len(data)
            temporal_qe = temporal_err / len(data)

            self.q_error_history.append(float(static_qe))
            self.temporal_q_error_history.append(float(temporal_qe))

            # restore training context state
            self.context_vector = old_context_vector

            """
            delta_main = cp.abs(self.weights - old_main_weights)
            self.avg_adjust_main.append(float(cp.mean(delta_main)))
            delta_context = cp.abs(self.context_weights - old_context_weights)
            self.avg_adjust_context.append(float(cp.mean(delta_context)))

            self.context_norms.append(cp.linalg.norm(self.context_vector))
            """

            # best checkpointing (keeps a copy of the best model state we have seen so far, 
            # instead of assuming the last epoch is the best)
            if temporal_qe < best_temporal_qe - min_delta:
                best_temporal_qe = temporal_qe
                self.best_epoch = epoch
                wait = 0

                best_weights = self.weights.copy()
                best_context_weights = self.context_weights.copy()
                best_context_vector = self.context_vector.copy()
            else:
                wait += 1
            
            # early stopping
            if (epoch + 1) >= min_epochs and wait >= patience:
                break
        
        # we restore best checkpoint
        if best_weights is not None:
            self.weights = best_weights
            self.context_weights = best_context_weights
            self.context_vector = best_context_vector

        # Recompute BMU trajectory using restored best model
        self.bmu_trajectory = []
        eval_context = cp.zeros(Q, dtype=dtype)

        old_context_vector = self.context_vector
        self.context_vector = eval_context

        for x in data:
            _, y, bmu = self._compute_energy_and_activity(x)
            self.bmu_trajectory.append(bmu)
            self.context_vector = y

        # keep restored best context state if you want it back
        self.context_vector = best_context_vector if best_context_vector is not None else old_context_vector
