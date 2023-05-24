import numpy as np

def aple(f, y):
    """
    Logarithm of Evidence with Strong A Prior
    :param f: feature with size (N, D)
    :param y: label with size (N), each term in range(n_class)
    :return: log evidence
    """
    N = len(f)
    D = len(f[0])
    num_dim = int(y.max() + 1)
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
        s = s.reshape(-1, 1)
        s = s[:k]
        u = u[:, :k]
    # u.shape = N x k
    # s.shape = k
    # vh.shape = k x D
    s = s.reshape(-1, 1)
    sigma = (s ** 2)
    sigma_full = sigma
    if N < D:
        sigma_full = np.zeros((D, 1))
        sigma_full[:k] = sigma
    evidences = []
    for i in range(num_dim):

        y_ = (y == i).astype(np.float64)
        y_ = y_.reshape(-1, 1)
        z = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
        z2 = z ** 2
        delta = (y_ ** 2).sum() - z2.sum()  # if k < N, we compute sum of xi for 0 singular values directly


        t =  (sigma[0] / N)
        m2 = (sigma * z2 / ((t + sigma) ** 2)).sum()
        res2 = (z2 / ((1 + sigma / t) ** 2)).sum() + delta
        beta = N / (res2 + t * m2)
        alpha = t * beta
        sigma_full = np.zeros((D, 1))
        sigma_full[:k] = sigma
        evidence = D / 2.0 * np.log(alpha) \
                 + N / 2.0 * np.log(beta) \
                 - 0.5 * np.sum(np.log(alpha + beta * sigma_full)) \
                 - beta / 2.0 * res2 \
                 - alpha / 2.0 * m2 \
                 - N / 2.0 * np.log(2 * np.pi)
        evidences.append(evidence / N)

    return np.mean(evidences)