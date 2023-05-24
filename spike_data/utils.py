import torch.utils.data as data
import torch
import scipy.io as sio
import h5py
import time
import bisect
import numpy as np


def one_hot(label, n_classes):
    """
    convert label to 1-hot
    :param label: label of the sample, in range(n_classed)
    :param n_classes: number of classes to classify
    :return: 1-hot of label
    """
    out = np.zeros(n_classes)
    out[label] = 1
    return out


def events_to_frames(times, coordinates, dt=1000, ds=4, n_step=60, size=None):
    """
    convert events into frames
    :param times: time of each event in us
    :param coordinates: coordinate of each event in us
    :param dt: temporal resolution
    :param ds: spatial resolution
    :param n_step: number of time steps
    :param size: size of one frame, (channels, width, height)
    :return: frames (channels, width, height, n_step)
    """
    if size is None:
        size = [2, 32, 32]
    start_times = range(times[0], times[0] + n_step * dt, dt)  # start time of each frame
    frames = np.zeros(size + [n_step], dtype='int8')  # (channels, width, height, n_step)
    i_start = 0
    i_end = 0
    for i, t in enumerate(start_times):
        i_end += bisect.bisect_left(times[i_end:], t + dt)
        if i_end > i_start:
            coord = coordinates[i_start: i_end]
            p, x, y = coord[:, 2], coord[:, 0] // ds, coord[:, 1] // ds
            np.add.at(frames, (p, x, y, i), 1)
        i_start = i_end
    return frames
