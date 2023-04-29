import numpy as np
from LogME.LogME import *
import matplotlib.pyplot as plt


def sign(x, y):
    if x >= y:
        return 1
    if x < y:
        return -1


def relative_coefficient(score, rank):
    result = 0
    for i in range(len(score)):
        for j in range(len(score) - i - 1):
            result += sign(score[i], score[i + j + 1]) * sign(rank[i], rank[i + j + 1])
    return result * 2 / len(score) / (len(score) - 1)


if __name__ == '__main__':
    N = 10
    D = 100

    f = np.random.randn(N, D)
    exp_f = np.exp(f)
    f = exp_f/np.sum(exp_f, axis=1, keepdims=True)

    y = np.zeros((N,))
    for i in range(N):
        y[i] = i % 10

    f_2 = np.zeros((N * 2, D))
    y_2 = np.zeros((N * 2,))
    f_2[:N, :] = f
    f_2[N:, :] = f
    y_2[:N] = y
    y_2[N:] = y
    logme = LogME()
    print(logme.fit(f, y))
    logme = LogME()
    print(logme.fit(f_2, y_2))
    # u, lam, uh = np.linalg.svd(f @ f.transpose())
    # v, lam2, vh = np.linalg.svd(f_2 @ f_2.transpose())
    # print(233)
