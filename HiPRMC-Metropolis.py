import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

N = 25  # size of image (NxN)
disc, rand_start, DFT = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
initial = np.random.random((N, N))
T = 1.0

# This bit creates a disc to try and model via RMC
for i in range(0, N):
    for j in range(0, N):
        r = (i - N / 2) ** 2 + (j - N / 2) ** 2
        if r < (N / 2 - 2) ** 2 and r > (N / 2 - 4) ** 2:
            disc[i, j] = 1
        else:
            disc[i, j] = 0

# plt.imshow(disc, cmap='Greys', origin="lower")
# plt.axis('off')
# plt.show()

rand_start[initial > np.count_nonzero(disc == 1) / N / N] = [0]
rand_start[initial < np.count_nonzero(disc == 1) / N / N] = [1]

def fourier_transform(m):
    return fftpack.fft2(m)

def chi_square(simulation, image):
    chi_2 = 0.0
    for i in range(0, N):
        for j in range(0, N):
            chi_2 += (abs(simulation[i][j] - image[i][j]) ** 2) / abs(image[i][j])
    return chi_2


def DFT_Matrix(x_old, y_old, x_new, y_new, simulated_image):
    U_new = np.outer(simulated_image[x_new, :], simulated_image[:, y_new])
    U_old = np.outer(simulated_image[x_old, :], simulated_image[:, y_old])
    U = U_new - U_old
    return U


def Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi, simulated_image):
    global acceptance
    if delta_chi < 0:
        simulated_image[x_old][y_old] = new_point
        simulated_image[x_new][y_new] = old_point
        acceptance += 1.0
    else:
        b = np.exp(-delta_chi / T)
        if b > np.random.rand():
            simulated_image[x_old][y_old] = new_point
            simulated_image[x_new][y_new] = old_point
            acceptance += 1.0
        else:
            simulated_image[x_old][y_old] = old_point
            simulated_image[x_new][y_new] = new_point


simulated_image = rand_start.copy()
F_image = fourier_transform(disc)
counter = 0
for t in range(0, 50):
    move_count = 0.0
    acceptance = 0.0
    for i in range(0, N):
        for j in range(0, N):
            move_count += 1
            F_old = fourier_transform(simulated_image)
            chi_old = chi_square(F_old, F_image)
            x_old = np.random.randint(0, N)
            y_old = np.random.randint(0, N)
            x_new = np.random.randint(0, N)
            y_new = np.random.randint(0, N)
            old_point = simulated_image[x_old][y_old]
            new_point = simulated_image[x_new][y_new]
            while new_point == old_point:
                x_new = np.random.randint(0, N)
                y_new = np.random.randint(0, N)
                new_point = simulated_image[x_new][y_new]
            F_new = fourier_transform(simulated_image) + DFT_Matrix(x_old, y_old, x_new, y_new, simulated_image)
            chi_new = chi_square(F_new, F_image)
            delta_chi = chi_new - chi_old
            Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi, simulated_image)
    print("The Acceptance Rate is", acceptance / move_count)

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(rand_start, cmap="Greys", origin="lower")
plt.title("original")
f.add_subplot(1, 2, 2)
plt.imshow(simulated_image, cmap="Greys", origin="lower")
plt.title("updated")
plt.show()

# plt.imshow(disc, cmap='Greys', origin="lower")
# plt.axis('off')
# plt.show()
# plt.imshow(np.log10(abs(fourier_transform(disc))), origin='lower', cmap='Greys')
# plt.axis('off')
# plt.show()
