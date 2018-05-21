import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from math import sqrt

disc, rand_start = np.zeros((256, 256)), np.zeros((256, 256))
initial = np.random.random((256, 256))

for i in range(0, 256):
    for j in range(0, 256):
        r = (i - 128) ** 2 + (j - 128) ** 2
        if r < 126 ** 2 and r > 122 ** 2:
            disc[i, j] = 2
        else:
            disc[i, j] = 1

rand_start[initial > np.count_nonzero(disc == 1) / 256 / 256] = [1]
rand_start[initial < np.count_nonzero(disc == 1) / 256 / 256] = [2]


def fourier_transform(m):
    return fftpack.fft2(m)


def chi_square(simulation, image):
    chi_2 = 0.0
    for i in range(0, 256):
        for j in range(0, 256):
            chi_2 += (abs(simulation[i][j] - image[i][j]) ** 2) / image[i][j]
    return chi_2

# plt.imshow(disc, cmap='Greys', origin="lower")
# plt.axis('off')
# plt.show()
# plt.imshow(np.log10(abs(fourier_transform(disc))), origin='lower', cmap='Greys')
# plt.axis('off')
# plt.show()
