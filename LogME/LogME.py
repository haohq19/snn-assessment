import warnings
import numpy as np
from numba import njit


# @njit
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """

    alpha = 1.0
    beta = 1.0
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    while True:
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + alpha / beta)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        new_alpha = gamma / alpha_de
        delta = (y_ - fh @ m)
        beta_de = (delta ** 2).sum()
        new_beta = (N - gamma) / beta_de
        if np.abs(new_alpha - alpha) / alpha < 1e-3 and np.abs(new_beta - beta) / beta < 1e-3:
            break
        alpha = new_alpha
        beta = new_beta
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * beta_de \
               - alpha / 2.0 * alpha_de \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N


class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the ICML 2021 paper
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        at http://proceedings.mlr.press/v139/you21b.html
        """
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)
        s[s < 1e-10] = 0
        self.sigma_icml = s
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        evidences = []
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence = each_evidence(y_, f, fh, v, s, vh, N, D)
        evidences.append(evidence)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        N, D = f.shape  # k = min(N, D)
        if N > D:  # direct SVD may be expensive
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
            # if k == N:
            #     k -= 1
            s = s.reshape(-1, 1)
            s = s[:k]
            u = u[:, :k]
        # u.shape = N x k
        # s.shape = k
        # vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
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
        :param f: [N, F], feature matrix from pre-trained model
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

        return self._fit_fixed_point(f, y)   #  _fit_icml(f, y) #
