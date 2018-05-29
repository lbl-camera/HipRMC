import numpy as np
from scipy import fftpack
from operator import mul
import matplotlib.pyplot as plt
import progressbar

def fourier_transform(m):
    return fftpack.fft2(m)


def chi_square(ft_simulation, ft_image):
    return np.sum((np.square(abs(abs(ft_simulation) - abs(ft_image))) / abs(ft_image)))


# def DFT_Matrix(x_old, y_old, x_new, y_new, F_old, simulated_image):
#     before_row, before_column = F_old[x_old, :], F_old[:, y_old]
#     F_s = fourier_transform(simulated_image)
#     after_row, after_column = F_s[x_new, :], F_s[:, y_new]
#     U_new = np.outer(after_column, after_row)
#     U_old = np.outer(before_column, before_row)
#     U = U_new - U_old
#     return U


def Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi, simulated_image, T, acceptance, chi_old,
               chi_new):
    if delta_chi < 0:
        simulated_image[x_old][y_old] = new_point
        simulated_image[x_new][y_new] = old_point
        acceptance += 1.0
        chi_old = chi_new
    else:
        b = np.exp(-delta_chi / T)
        if b > np.random.rand():
            simulated_image[x_old][y_old] = new_point
            simulated_image[x_new][y_new] = old_point
            acceptance += 1.0
            chi_old = chi_new
        else:
            simulated_image[x_old][y_old] = old_point
            simulated_image[x_new][y_new] = new_point
    return acceptance, chi_old

def random_initial(image:np.array):
    initial = np.zeros_like(image)
    initial2 = initial.reshape(mul(*image.shape))
    initial2[: np.count_nonzero(image == 1)] = 1
    np.random.shuffle(initial2)
    initial = initial2.reshape(image.shape)
    return initial

def rmc(image:np.array, T, initial:np.array=None):
    if initial is None:
        initial = random_initial(image)

    simulated_image = initial.copy()
    F_image = fourier_transform(image)
    F_old = fourier_transform(simulated_image)
    chi_old = chi_square(F_old, F_image)
    for _ in progressbar.progressbar(range(0, 30000)):
        move_count = 0.0
        acceptance = 0.0
        for _ in range(np.count_nonzero(simulated_image == 1)):
            move_count += 1
            # before_move = simulated_image.copy()
            x_old = np.random.randint(0, image.shape[0])
            y_old = np.random.randint(0, image.shape[1])
            while simulated_image[x_old][y_old] != 1:
                x_old = np.random.randint(0, image.shape[0])
                y_old = np.random.randint(0, image.shape[1])
            x_new = np.random.randint(0, image.shape[0])
            y_new = np.random.randint(0, image.shape[1])
            old_point = simulated_image[x_old][y_old]
            new_point = simulated_image[x_new][y_new]
            while new_point == old_point:
                x_new = np.random.randint(0, image.shape[0])
                y_new = np.random.randint(0, image.shape[1])
                new_point = simulated_image[x_new][y_new]
            simulated_image[x_old][y_old] = new_point
            simulated_image[x_new][y_new] = old_point
            old_array = np.zeros_like(image)
            old_array[x_old][y_old] = 1
            new_array = np.zeros_like(image)
            new_array[x_new][y_new] = 1
            # U = DFT_Matrix(x_old, y_old, x_new, y_new, F_old, simulated_image)
            F_new = F_old - fourier_transform(old_array) + fourier_transform(new_array)
            chi_new = chi_square(F_new, F_image)
            delta_chi = chi_new - chi_old
            acceptance, chi_old = Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
                                             simulated_image, T, acceptance, chi_old, chi_new)
            F_old = fourier_transform(simulated_image)
        T = T * 0.9998

    return simulated_image
