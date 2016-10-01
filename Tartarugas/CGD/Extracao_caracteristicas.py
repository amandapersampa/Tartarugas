from setuptools.command.rotate import rotate
from skimage.feature import local_binary_pattern
from scipy.misc import imread
from time import time
import numpy as np

def minimum_value(original):
    aux = original
    rotationed = rotate(original)
    max = 9
    i = 0
    while (i < max):
        if (rotationed < aux):
            aux = rotationed
        rotationed = rotate(rotationed)
        i = i + 1
    return aux

def rotate(value):
    if (value % 2 == 0):
        return value >> 1
    else:
        value = value >> 1
    return value + 256

def lbp_rotation_invariant(img):
    METHOD = 'uniform'
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, METHOD).astype(np.int64)
    for i in range(len(lbp)):
        for j in range(len(lbp)):
            lbp[i][j] = minimum_value(lbp[i][j])
    return lbp

def histograma(channel):
    histogram, bin_edges = np.histogram(channel)
    hist = histogram
    hist = hist.tolist()
    return hist

def lbp(nome, m):
    tart = imread(nome, mode=m)
    channel_0 = lbp_rotation_invariant(tart[:, :, 0])
    hist = histograma(channel_0)
    channel_1 = lbp_rotation_invariant(tart[:, :, 1])
    hist.extend(histograma(channel_1))
    channel_2 = lbp_rotation_invariant(tart[:, :, 2])
    hist.extend(histograma(channel_2))
    return hist

def lbp_normal(nome, m):
    tart = imread(nome, mode=m)
    METHOD = 'uniform'
    radius = 3
    n_points = 8 * radius

    channel_0 = local_binary_pattern(tart[:, :, 0], n_points, radius, METHOD).astype(np.int64)
    hist = histograma(channel_0)

    channel_1 = local_binary_pattern(tart[:, :, 1], n_points, radius, METHOD).astype(np.int64)
    hist.extend(histograma(channel_1))

    channel_2 = local_binary_pattern(tart[:, :, 2], n_points, radius, METHOD).astype(np.int64)
    hist.extend(histograma(channel_2))
    return hist

def  characteristic_matrix_lbp_RGB (nome, rotation_inv=True) :
    return characteristic_matrix_lbp(nome, 'RGB',rotation_inv)

def  characteristic_matrix_lbp_YCbCr(nome, rotation_inv=True) :
    return characteristic_matrix_lbp(nome, 'YCbCr', rotation_inv)

def characteristic_matrix_lbp(nome, tipo,rotation_inv):
    if (rotation_inv):
        return lbp(nome, tipo)
    else:
        return lbp_normal(nome, tipo)

