import cupy as cp
from .SOM_vectorized import SOM_vectorized
DEFAULT_DTYPE = cp.float32

class RSOM(SOM_vectorized):
    def __init__(self,
                 m: int,
                 n: int,
                 dim: int,
                 alpha: float = 0.1, # controls how much the input matters relative to memory
                 beta: float = 0.5, # scaling factor for recurrent (context) distance term
                 *,
                 weight_init_method='uniform',
                 grid_metric='euclid',
                 neighborhood_kernel='gaussian',
                 lr0=0.08,
                 lr_final=0.005,
                 radius0=None,
                 radius_final=0.8,
                 lr_schedule=None,
                 radius_schedule=None,
                 seed=None
                 ):
        
        super().__init__(m=m, n=n, dim=dim,
                         weight_init_method=weight_init_method,
                         grid_metric=grid_metric,
                         neighborhood_kernel=neighborhood_kernel,
                         lr0=lr0,
                         lr_final=lr_final,
                         radius0=radius0,
                         radius_final=radius_final,
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
        self.bmu_trajectory = []          # flat
        self.bmu_trajectories = []        # per-sequence
        self.activity_trajectories = []   # per-sequence
        self.sequence_lengths = []
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

    
    def reset_context(self):
        Q = self.m * self.n
        self.context_vector = cp.zeros(Q, dtype=self.weights.dtype)


    def _store_sequence_traces(self, sequences):
        Q = self.m * self.n
        dtype = self.weights.dtype

        self.bmu_trajectory = []
        self.bmu_trajectories = []
        self.activity_trajectories = []
        self.sequence_lengths = []

        for seq in sequences:
            self.reset_context()

            seq_bmus = []
            seq_activities = []

            for x in seq:
                _, y, bmu = self._compute_energy_and_activity(x)
                # Convert BMU coordinates to plain Python ints for easy storage/use
                bmu_tuple = (int(bmu[0]), int(bmu[1]))
                seq_bmus.append(bmu_tuple)
                self.bmu_trajectory.append(bmu_tuple)
                # Save a copy so later updates to context don't overwrite history
                seq_activities.append(y.copy())
                self.context_vector = y

            self.bmu_trajectories.append(seq_bmus)
            self.sequence_lengths.append(len(seq_bmus))

            if len(seq_activities) > 0:
                self.activity_trajectories.append(cp.stack(seq_activities, axis=0))
            else:
                self.activity_trajectories.append(cp.empty((0, Q), dtype=dtype))

        self.reset_context()

    
    def train(self, 
              data, 
              num_epochs: int = 100,
              min_epochs: int = 75,
              patience: int = 8,
              min_delta: float = 1e-4
              ):
        # Accept:
        #   (T, dim)            -> one sequence
        #   (N, T, dim)         -> many equal-length sequences
        #   list of (Ti, dim)   -> many variable-length sequences

        if isinstance(data, (list, tuple)):
            sequences = [cp.asarray(seq, dtype=cp.float32) for seq in data]
        else:
            data = cp.asarray(data, dtype=cp.float32)
            if data.ndim == 2:
                sequences = [data]
            elif data.ndim == 3:
                sequences = [data[i] for i in range(data.shape[0])]
            else:
                raise ValueError("Expected data shape (T, dim), (N, T, dim), or list of sequences.")

        all_points = cp.concatenate(sequences, axis=0)
        self.init_weights(all_points)
        Q = self.m * self.n
        dtype = self.weights.dtype
        """
        You can also initialize the recurrent weights in a more stable manner. Rand in [0, 1] is okay, but many implementations
        use small values around 0 so you can do:
        self.context_weights = 0.01 * cp.random.randn(Q, Q, dtype=dtype)
        """
        self.context_weights = (0.01 * cp.random.randn(Q, Q)).astype(dtype)
        self.context_vector = cp.zeros(Q, dtype=dtype)

        self.q_error_history = []
        self.temporal_q_error_history = []
        # self.avg_adjust_main = []
        # self.avg_adjust_context = []
        # self.context_norms = []
        self.bmu_trajectory = []
        # self.context_history = []

        # early stopping / checkpointing
        best_temporal_qe = float("inf")
        self.best_epoch = -1
        wait = 0

        best_weights = None
        best_context_weights = None

        for epoch in range(num_epochs):
            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)

            # old_main_weights = self.weights.copy()
            # old_context_weights = self.context_weights.copy()

            order = cp.random.permutation(len(sequences))

            for seq_idx in order:
                seq = sequences[int(seq_idx)]
                self.reset_context()
                for x in seq:
                    d_comb, y, bmu = self._compute_energy_and_activity(x)
                    self.update_weights(x, bmu, lr, radius)
                    self.context_vector = y  # Update global context to current activity

            # evaluation
            static_err = 0.0
            temporal_err = 0.0
            total_steps = 0

            for seq in sequences:
                self.reset_context()
                for x in seq:
                    d_comb, y, bmu = self._compute_energy_and_activity(x)

                    # input-only error
                    w_bmu = self.weights[bmu]
                    static_err += cp.linalg.norm(x - w_bmu)

                    # temporal error
                    bmu_idx = bmu[0] * self.n + bmu[1]
                    temporal_err += cp.sqrt(d_comb[bmu_idx])
                    
                    # advance recurrent state during evaluation
                    self.context_vector = y
                    total_steps += 1
            

            static_qe = static_err / total_steps
            temporal_qe = temporal_err / total_steps

            self.q_error_history.append(float(static_qe))
            self.temporal_q_error_history.append(float(temporal_qe))

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
            else:
                wait += 1
            
            # early stopping
            if (epoch + 1) >= min_epochs and wait >= patience:
                break
        
        # we restore best checkpoint
        if best_weights is not None:
            self.weights = best_weights
            self.context_weights = best_context_weights
        else:
            self.best_epoch = num_epochs
        

        # Recompute BMU trajectory using restored best model
        self._store_sequence_traces(sequences)
