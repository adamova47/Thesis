import cupy as cp
from .SOM_vectorized import SOM_vectorized

DEFAULT_DTYPE = cp.float32

class MSOM(SOM_vectorized):
    def __init__(self, 
                 m, 
                 n, 
                 dim,
                 alpha: float = 0.2,
                 beta: float = 0.5,
                 *, 
                 weight_init_method = "uniform", 
                 grid_metric = "euclid", 
                 neighborhood_kernel = "gaussian",
                 lr0=0.08,
                 lr_final=0.005,
                 radius0=None,
                 radius_final=0.8, 
                 lr_schedule = None, 
                 radius_schedule = None, 
                 seed=None,
                 context_init="zeros",  # "zeros" or "random_small"
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
        self.context_init = context_init

        self.avg_adjust_main = []
        self.avg_adjust_context = []
        self.bmu_trajectory = []
        self.context_descriptor_history = []
        self.temporal_q_error_history = []

    def _init_context_weights(self):
        dtype = self.weights.dtype
        if self.context_init == "zeros":
            self.context_weights = cp.zeros((self.m, self.n, self.dim), dtype=dtype)
        elif self.context_init == "random_small":
            self.context_weights = (0.01 * cp.random.randn(self.m, self.n, self.dim)).astype(dtype)
        else:
            raise ValueError("context_init must be 'zeros' or 'random_small'")

    def _compute_context_descriptor(self, prev_bmu):
        """
        Compute C_t from previous winner:
        C_t = (1-beta) * w_prev + beta * c_prev
        If prev_bmu is None (sequence start), use 0 vector (matches thesis assumption).
        """
        if prev_bmu is None:
            return cp.zeros(self.dim, dtype=self.weights.dtype)
        
        w_prev = self.weights[prev_bmu]
        c_prev = self.context_weights[prev_bmu]
        return (1.0 - self.beta) * w_prev + self.beta * c_prev
    
    def find_bmu(self, x, C_t):
        """
        BMU under merged distance:
        d = (1-alpha)||x-w||^2 + alpha||C_t-c||^2
        Vectorized over (m, n).
        """
        diff_x = self.weights - x
        d_x = cp.einsum("ijk,ijk->ij", diff_x, diff_x) # (m, n)

        # context term
        diff_c = self.context_weights - C_t
        d_c = cp.einsum("ijk,ijk->ij", diff_c, diff_c)

        d = (1.0 - self.alpha) * d_x + self.alpha * d_c
        min_index = cp.argmin(d)
        return (min_index // self.n, min_index % self.n)

    def update_weights(self, x, C_t, bmu, lr, radius):
        """
        SOM neighborhood update for both weights and contexts:
            w_i <- w_i + lr * h_ib * (x   - w_i)
            c_i <- c_i + lr * h_ib * (C_t - c_i)
        """
        i0, j0 = bmu
        dists = self.grid_distance(i0, j0)
        h = self.compute_neighborhood(dists, radius).astype(self.weights.dtype) # (m, n)

        self.weights += lr * h[:, :, None] * (x - self.weights)
        self.context_weights += lr * h[:, :, None] * (C_t - self.context_weights)

    def train(
        self,
        data,
        num_epochs: int = 100,
        min_epochs: int = 100,
        patience: int = 5,
        min_delta: float = 1e-4
    ):
        # Accept:
        #   (T, dim)            -> one sequence
        #   (N, T, dim)         -> many equal-length sequences
        #   list of (Ti, dim)   -> many variable-length sequences

        if isinstance(data, (list, tuple)):
            sequences = [cp.asarray(seq, dtype=DEFAULT_DTYPE) for seq in data]
        else:
            data = cp.asarray(data, dtype=DEFAULT_DTYPE)
            if data.ndim == 2:
                sequences = [data]
            elif data.ndim == 3:
                sequences = [data[i] for i in range(data.shape[0])]
            else:
                raise ValueError("Expected data shape (T, dim), (N, T, dim), or list of sequences.")

        all_points = cp.concatenate(sequences, axis=0)
        self.init_weights(all_points)
        self._init_context_weights()

        self.q_error_history = []
        self.temporal_q_error_history = []
        self.bmu_trajectory = []
        self.bmu_trajectories = []
        # self.avg_adjust_main = []
        # self.avg_adjust_context = []
        # self.context_descriptor_history = []
        self.sequence_lengths = []
        self.best_epoch = -1

        # early stopping / checkpointing
        best_temporal_qe = float("inf")
        wait = 0
        best_weights = None
        best_context_weights = None

        for epoch in range(num_epochs):
            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)

            # old_w = self.weights.copy()
            # old_c = self.context_weights.copy()

            order = cp.random.permutation(len(sequences))

            # training pass
            for seq_idx in order:

                seq = sequences[int(seq_idx)]
                
                prev_bmu = None
                # sequence boundaries should reset temporal state
                # even if later you decide to use some cross-epoch carryover

                for x in seq:
                    C_t = self._compute_context_descriptor(prev_bmu)
                    bmu = self.find_bmu(x, C_t)

                    self.update_weights(x, C_t, bmu, lr, radius)

                    prev_bmu = bmu

            # post-epoch evaluation pass
            static_err = 0.0
            temporal_err = 0.0
            total_steps = 0

            for seq in sequences:
                eval_prev_bmu = None

                for x in seq:
                    C_t = self._compute_context_descriptor(eval_prev_bmu)
                    bmu = self.find_bmu(x, C_t)

                    w_bmu = self.weights[bmu]
                    c_bmu = self.context_weights[bmu]

                    # input-only error
                    static_err += cp.linalg.norm(x - w_bmu)

                    # temporal error
                    d_x = cp.sum((x - w_bmu) ** 2)
                    d_c = cp.sum((C_t - c_bmu) ** 2)
                    temporal_err += cp.sqrt((1.0 - self.alpha) * d_x + self.alpha * d_c)

                    eval_prev_bmu = bmu
                    total_steps += 1

            static_qe = static_err / total_steps
            temporal_qe = temporal_err / total_steps

            self.q_error_history.append(float(static_qe))
            self.temporal_q_error_history.append(float(temporal_qe))

            """self.avg_adjust_main.append(cp.mean(cp.abs(self.weights - old_w)))
            self.avg_adjust_context.append(cp.mean(cp.abs(self.context_weights - old_c)))"""

            # best checkpointing
            if temporal_qe < best_temporal_qe - min_delta:
                best_temporal_qe = float(temporal_qe)
                self.best_epoch = epoch
                wait = 0
                best_weights = self.weights.copy()
                best_context_weights = self.context_weights.copy()
            else:
                wait += 1

            # early stopping
            if (epoch + 1) >= min_epochs and wait >= patience:
                break

        # restore best checkpoint
        if best_weights is not None:
            self.weights = best_weights
            self.context_weights = best_context_weights
        else:
            self.best_epoch = num_epochs - 1

        # rebuild BMU trajectories using restored best model
        self.bmu_trajectory = []
        self.bmu_trajectories = []
        self.sequence_lengths = []

        for seq in sequences:
            prev_bmu = None
            seq_bmus = []

            for x in seq:
                C_t = self._compute_context_descriptor(prev_bmu)
                bmu = self.find_bmu(x, C_t)

                bmu_tuple = (int(bmu[0]), int(bmu[1]))
                self.bmu_trajectory.append(bmu_tuple)
                seq_bmus.append(bmu_tuple)

                # self.context_descriptor_history.append(C_t.copy())

                prev_bmu = bmu

            self.bmu_trajectories.append(seq_bmus)
            self.sequence_lengths.append(len(seq_bmus))