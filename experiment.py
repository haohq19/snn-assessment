import numpy as np
from LogME.LogME import *
import matplotlib.pyplot as plt
import math

def test(f, y):
    N = len(f)
    D = len(f[0])
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
    s_mut = np.exp(1/k * np.sum(np.log(s)))
    sigma_full = sigma
    if N < D:
        sigma_full = np.zeros((D, 1))
        sigma_full[:k] = sigma
    evidences = []

    alpha, beta = 1.0, 1.0
    for i in range(num_dim):

        y_ = (y == i).astype(np.float64)
        y_ = y_.reshape(-1, 1)
        z = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
        z2 = z ** 2
        delta = (y_ ** 2).sum() - z2.sum()  # if k < N, we compute sum of xi for 0 singular values directly

        # for _ in range(101):
        #     gamma = (sigma / (sigma + alpha / beta)).sum()
        #     m2 = (sigma * z2 / ((alpha / beta + sigma) ** 2)).sum()
        #     res2 = (z2 / ((1 + sigma / (alpha / beta)) ** 2)).sum() + delta
        #     new_alpha = gamma / m2
        #     new_beta = (N - gamma) / res2
        #     if np.abs((new_alpha - alpha) / alpha) < 1e-3 and np.abs((new_beta - beta) / beta) < 1e-3:
        #         break
        #     alpha = new_alpha
        #     beta = new_beta
        # sigma_full = np.zeros((D, 1))
        # sigma_full[:k] = sigma
        # evidence = D / 2.0 * np.log(alpha) \
        #            + N / 2.0 * np.log(beta) \
        #            - 0.5 * np.sum(np.log(alpha + beta * sigma_full)) \
        #            - beta / 2.0 * res2 \
        #            - alpha / 2.0 * m2 \
        #            - N / 2.0 * np.log(2 * np.pi)
        # evidences.append(evidence / N)

        t = 1 / N * sigma[0]
        # t = (D / N) # * np.std(sigma)
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

if __name__ == '__main__':
    N = 10
    D = 100
    num_dim = 10

    # f = np.random.randn(N, D)
    # exp_f = np.exp(f)
    # f = exp_f/np.sum(exp_f, axis=1, keepdims=True)
    y = np.zeros((N,))
    for i in range(N):
        y[i] = i % num_dim

    f = np.zeros([N, D])
    noise = np.random.randn(N, D)
    for i in range(N):
        for j in range(int(D/num_dim)):
            f[i][i % num_dim + j * num_dim] = 1

    snrs = np.arange(0.1, 1, 0.1)
    evidences = np.zeros(len(snrs))
    evidences2 = np.zeros(len(snrs))
    i = 0
    for snr in snrs:
        _f = f + snr * noise
        # _f = np.exp(_f)
        # _f = _f / np.sum(_f * _f, axis=1, keepdims=True)
        # _f = _f / np.sqrt(np.sum(_f * _f, axis=1, keepdims=True))
        evidences[i] = test(_f, y)
        logme = LogME()
        evidences2[i] = logme.fit(_f, y)
        i += 1
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.plot(snrs, evidences, label='fake')
    plt.plot(snrs, evidences2, label='logme')
    plt.legend()
    plt.show()

    # logme = LogME()
    # score = logme.fit(f, y)
    # print(score)

    # if N > D:  # direct SVD may be expensive
    #     v, lam, vh = np.linalg.svd(f.transpose() @ f)
    #     s = np.sqrt(lam)
    #     u_times_sigma = f @ vh.transpose()
    #     k = np.sum((s > 1e-10) * 1)  # rank of f
    #     s = s.reshape(-1, 1)
    #     s = s[:k]
    #     u = u_times_sigma[:, :k] / s.reshape(1, -1)
    # else:  # N <= D
    #     u, lam, uh = np.linalg.svd(f @ f.transpose())
    #     s = np.sqrt(lam)
    #     k = np.sum((s > 1e-10) * 1)  # rank of f
    #     s = s.reshape(-1, 1)
    #     s = s[:k]
    #     u = u[:, :k]
    # # u.shape = N x k
    # # s.shape = k
    # # vh.shape = k x D
    # s = s.reshape(-1, 1)
    # sigma = (s ** 2)
    # sigma_full = sigma
    # if N < D:
    #     sigma_full = np.zeros((D, 1))
    #     sigma_full[:k] = sigma
    #
    #
    # alphas = np.arange(0, 5, 0.1)
    # betas = np.arange(5, 15, 0.2)
    # evidences = []
    # evidences2 = []
    # evidences3 = []

    # for i in range(num_dim):
    #     # L = np.zeros((len(alphas), len(betas)))
    #     # for index_alpha in range(len(alphas)):
    #     #     for index_beta in range(len(betas)):
    #     #         _alpha = alphas[index_alpha]
    #     #         _beta = betas[index_beta]
    #     #         alpha = 10 ** _alpha
    #     #         beta = _beta
    #     #         evidences = []
    #     #
    #     #         y_ = (y == i).astype(np.float64)
    #     #         y_ = y_.reshape(-1, 1)
    #     #         z = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
    #     #         z2 = z ** 2
    #     #         delta = (y_ ** 2).sum() - z2.sum()  # if k < N, we compute sum of xi for 0 singular values directly
    #     #
    #     #         m2 = (sigma * z2 / ((alpha / beta + sigma) ** 2)).sum()
    #     #         res2 = (z2 / ((1 + sigma / (alpha / beta)) ** 2)).sum() + delta
    #     #
    #     #         evidence = D / 2.0 * np.log(alpha) \
    #     #                    + N / 2.0 * np.log(beta) \
    #     #                    - 0.5 * np.sum(np.log(alpha + beta * sigma_full)) \
    #     #                    - beta / 2.0 * res2 \
    #     #                    - alpha / 2.0 * m2 \
    #     #                    - N / 2.0 * np.log(2 * np.pi)
    #     #         L[index_alpha, index_beta] = evidence / N
    #     # L = L.transpose()
    #     #
    #     # fig = plt.figure(figsize=(16, 16))
    #     # X, Y = np.meshgrid(alphas, betas)
    #     # plt.title("log evidence, N={}, D={}".format(N, D), fontsize=20)
    #     # cont = plt.contour(X, Y, L, 128, colors='black')
    #     # plt.contourf(X, Y, L, 128)
    #     # plt.clabel(cont, inline=1, fontsize=10)
    #     # plt.show()
    #
    #
    # # surf = ax.plot_surface(alphas, betas, L, cstride=1, cmap=plt.get_cmap('rainbow'))
    # # plt.show()
    #
    #
    #     y_ = (y == i).astype(np.float64)
    #     y_ = y_.reshape(-1, 1)
    #     z = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
    #     z2 = z ** 2
    #     delta = (y_ ** 2).sum() - z2.sum()  # if k < N, we compute sum of xi for 0 singular values directly

        # alpha = 1.0
        # beta = 1.0
        # count = 0
        # res2 = 0
        # m2 = 0
        # for _ in range(8):
        #     m2 = (sigma * z2 / ((alpha / beta + sigma) ** 2)).sum()
        #     res2 = (z2 / ((1 + sigma / (alpha / beta)) ** 2)).sum() + delta
        #     new_alpha = (D + 1) / m2
        #     new_beta = (N + 1) / res2
        #     if np.abs((new_alpha - alpha) / alpha) < 1e-3 and np.abs((new_beta - beta) / beta) < 1e-3:
        #         break
        #     alpha = new_alpha
        #     beta = new_beta
        #     #  log_likelihood = - D/2 * np.log(m2) - N/2 * np.log(res2)
        #
        # sigma_full = np.zeros((D, 1))
        # sigma_full[:k] = sigma
        # evidence = D / 2.0 * np.log(alpha) \
        #            + N / 2.0 * np.log(beta) \
        #            - 0.5 * np.sum(np.log(alpha + beta * sigma_full)) \
        #            - beta / 2.0 * res2 \
        #            - alpha / 2.0 * m2 \
        #            - N / 2.0 * np.log(2 * np.pi)
        # evidences.append(evidence/N)
        # print(evidence)
        #
        #
        # alpha, beta = 1.0, 1.0
        # for _ in range(8):
        #     gamma = (sigma / (sigma + alpha / beta)).sum()
        #     m2 = (sigma * z2 / ((alpha / beta + sigma) ** 2)).sum()
        #     res2 = (z2 / ((1 + sigma / (alpha / beta)) ** 2)).sum() + delta
        #     new_alpha = gamma / m2
        #     new_beta = (N - gamma) / res2
        #     if np.abs((new_alpha - alpha) / alpha) < 1e-3 and np.abs((new_beta - beta) / beta) < 1e-3:
        #         break
        #     alpha = new_alpha
        #     beta = new_beta
        # sigma_full = np.zeros((D, 1))
        # sigma_full[:k] = sigma
        # evidence2 = D / 2.0 * np.log(alpha) \
        #            + N / 2.0 * np.log(beta) \
        #            - 0.5 * np.sum(np.log(alpha + beta * sigma_full)) \
        #            - beta / 2.0 * res2 \
        #            - alpha / 2.0 * m2 \
        #            - N / 2.0 * np.log(2 * np.pi)
        # evidences2.append(evidence2 / N)
        # print(evidence2)


    #     t = D / N
    #     gamma = (sigma / (sigma + t)).sum()
    #     m2 = (sigma * z2 / ((t + sigma) ** 2)).sum()
    #     res2 = (z2 / ((1 + sigma / t) ** 2)).sum() + delta
    #     beta = N / (res2 + t * m2)
    #     alpha = t * beta
    #     beta_argmax = beta
    #     sigma_full = np.zeros((D, 1))
    #     sigma_full[:k] = sigma
    #     evidence3 = D / 2.0 * np.log(alpha) \
    #               + N / 2.0 * np.log(beta) \
    #               - 0.5 * np.sum(np.log(alpha + beta * sigma_full)) \
    #               - beta / 2.0 * res2 \
    #               - alpha / 2.0 * m2 \
    #               - N / 2.0 * np.log(2 * np.pi)
    #     evidences3.append(evidence3 / N)
    # print(np.mean(evidences3))

        # betas = np.arange(0, 5, 0.01)
        # betas = np.exp(betas)
        # evidence4 = np.zeros(len(betas))
        # for beta_index in range(len(betas)):
        #     beta = betas[beta_index]
        #     alpha = t * beta
        #     evidence4[beta_index] = D / 2.0 * np.log(alpha) \
        #                  + N / 2.0 * np.log(beta) \
        #                  - 0.5 * np.sum(np.log(alpha + beta * sigma_full)) \
        #                  - beta / 2.0 * res2 \
        #                  - alpha / 2.0 * m2 \
        #                  - N / 2.0 * np.log(2 * np.pi)
        #
        # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # ax.loglog(betas,  - evidence4 / N)
        # plt.show()