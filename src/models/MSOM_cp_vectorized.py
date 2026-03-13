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
                 lr_schedule = None, 
                 radius_schedule = None, 
                 seed=None,
                 context_init="zeros",  # "zeros" or "random_small"
                 ):
        super().__init__(m, n, dim, 
                         weight_init_method, 
                         grid_metric, 
                         neighborhood_kernel, 
                         lr_schedule, 
                         radius_schedule, 
                         seed)
        
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

    def train(self, data, num_epochs: int = 100, reset_context_each_epoch : bool = True):
        training_data = cp.asarray(data, dtype=DEFAULT_DTYPE)
        training_data = training_data.reshape(-1, self.dim)
        self.init_weights(training_data)
        self._init_context_weights()
        prev_bmu = None

        self.q_error_history = []
        # self.avg_adjust_main = []
        # self.avg_adjust_context = []
        self.bmu_trajectory = []
        # self.context_descriptor_history = []

        for epoch in range(num_epochs):
            lr = self.lr_schedule(epoch, num_epochs)
            radius = self.radius_schedule(epoch, num_epochs)

            # old_w = self.weights.copy()
            # old_c = self.context_weights.copy()

            if reset_context_each_epoch:
                prev_bmu = None

            
            # training pass
            for x in training_data:
                C_t = self._compute_context_descriptor(prev_bmu)
                bmu = self.find_bmu(x, C_t)

                self.update_weights(x, C_t, bmu, lr, radius)

                prev_bmu = bmu

                if epoch == num_epochs - 1:
                    self.bmu_trajectory.append(bmu)
                    # self.context_descriptor_history.append(C_t.copy())

            # post-epoch evaluation pass
            eval_prev_bmu = None if reset_context_each_epoch else prev_bmu

            static_err = 0.0
            temporal_err = 0.0

            for x in training_data:
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

            static_qe = static_err / len(training_data)
            temporal_qe = temporal_err / len(training_data)

            self.q_error_history.append(float(static_qe))
            self.temporal_q_error_history.append(float(temporal_qe))

            """self.avg_adjust_main.append(cp.mean(cp.abs(self.weights - old_w)))
            self.avg_adjust_context.append(cp.mean(cp.abs(self.context_weights - old_c)))"""