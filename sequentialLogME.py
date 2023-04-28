import numpy as np
from models.spike_func import *


class SequentialLogME(object):
    def __init__(self, decay=0.2, thresh=0.3, regression=False):
        """
            :param decay: decay of membrane potential
            :param regression: whether regression
        """
        self.decay = decay
        self.thresh = thresh
        self.regression = regression
        self.spike_func = SpikeFunc.apply

    def _fit(self, f: np.ndarray, y: np.ndarray):
        self.num_feat = f.shape[0]
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        self.num_step = f.shape[2]
        mem = spike = np.zeros((self.num_feat, self.num_step))
        for step in range(self.num_step):
            f_step = f[..., step]
            N, D = f_step.shape  # k = min(N, D)

            # compute SVD
            if N > D:
                v, lam, vh = np.linalg.svd(f.transpose() @ f)
                s = np.sqrt(lam)
                u_times_sigma = f @ vh.transpose()
                k = np.sum((s > 1e-10) * 1)  # rank of f
                s = s.reshape(-1, 1)
                s = s[:k]
                u = u_times_sigma[:, :k] / s.reshape(1, -1)
            else:  # N <= D
                u, lam, uh = np.linalg.svd(f @ f.transpose())
                s = np.sqrt(lam)
                k = np.sum((s > 1e-10) * 1)  # rank of f
                s = s.reshape(-1, 1)
                s = s[:k]
                u = u[:, :k]
            # u.shape = N x k
            # s.shape = k
            s = s.reshape(-1, 1)
            sigma = (s ** 2)


            counts = []
            evidences = []
            for i in range(self.num_dim):
                y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
                y_ = y_.reshape(-1, 1)
                z = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
                z2 = z ** 2
                delta = (y_ ** 2).sum() - z2.sum()  # if k < N, we compute sum of xi for 0 singular values directly

                alpha, beta = 1.0, 1.0
                count = 0
                while True:  # for _ in range(11):
                    count += 1
                    gamma = (sigma / (sigma + alpha / beta)).sum()
                    m2 = (sigma * z2 / ((alpha / beta + sigma) ** 2)).sum()
                    res2 = (z2 / ((1 + sigma / (alpha / beta)) ** 2)).sum() + delta
                    new_alpha = gamma / m2
                    new_beta = (N - gamma) / res2
                    if np.abs((new_alpha - alpha) / alpha) < 1e-3 and np.abs((new_beta - beta) / beta) < 1e-3:
                        break
                    alpha = new_alpha
                    beta = new_beta
                sigma_full = np.zeros((D, 1))
                sigma_full[:k] = sigma
                self.sigma_fp = sigma_full
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma_full)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)

                counts.append(count)
                evidences.append(evidence / N)
            # print(np.mean(counts))
            return np.mean(evidences)



    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F, N_Step], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels

        :return: LogME score (how well f can fit y directly)
        """

        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

        return self._fit(f, y)

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * self.decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike
