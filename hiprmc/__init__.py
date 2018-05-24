import numpy as np
from scipy import fftpack
from operator import mul
import matplotlib.pyplot as plt
import progressbar

def fourier_transform(m):
    return fftpack.fft2(m)

def chi_square(simulation, image):
    return np.sum((np.square(abs(simulation - image))/abs(image)))


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
    error, run_count, Temperature = [], [], []
    for t in progressbar.progressbar(range(0, 40000)):
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
            F_new = F_old - fftpack.fft2(old_array) + fftpack.fft2(new_array)
            chi_new = chi_square(F_new, F_image)
            delta_chi = chi_new - chi_old
            acceptance, chi_old = Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
                                             simulated_image, T, acceptance, chi_old, chi_new)
            F_old = fourier_transform(simulated_image)
        T = T * 0.9998
        error.append(chi_old)
        run_count.append(t)

    plt.plot(run_count, error, 'b-')
    plt.xlabel('Monte Carlo Iteration')
    plt.ylabel('Chi-Squared Value')
    plt.show()

    return simulated_image
