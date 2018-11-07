import numpy as np
from scipy import fftpack, optimize
from operator import mul
from hiprmc.temperature_tuning import temp_tuning
import hiprmc
import progressbar
import matplotlib.pyplot as plt

def fourier_transform(m):
    return fftpack.fft2(m)

def chi_square(ft_simulation, ft_image):
    return np.sum((np.square(abs(abs(ft_simulation) ** 2 - abs(ft_image) ** 2)) / np.linalg.norm(ft_image) ** 2))

# def DFT_Matrix(x_old, y_old, x_new, y_new, F_old, simulated_image):
#     before_row, before_column = F_old[x_old, :], F_old[:, y_old]
#     F_s = fourier_transform(simulated_image)
#     after_row, after_column = F_s[x_new, :], F_s[:, y_new]
#     U_new = np.outer(after_column, after_row)
#     U_old = np.outer(before_column, before_row)
#     U = U_new - U_old
#     return U

def Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi, simulated_image, T, acceptance, chi_old,
               chi_new, T_MAX, iter, N, t_step):
    if T <= 0:
        simulated_image[x_old][y_old] = old_point
        simulated_image[x_new][y_new] = new_point
    else:
        if delta_chi < 0:
            simulated_image[x_old][y_old] = new_point
            simulated_image[x_new][y_new] = old_point
            acceptance += 1.0
            chi_old = chi_new
        else:
            temperature = T / (T_MAX - T)
            b = np.exp(-delta_chi / temperature * np.count_nonzero(simulated_image == 1) * N ** 2 / (1 + t_step * iter))
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
    # initial_size = int(len(image) / 4)
    # initial = np.zeros((initial_size, initial_size))
    # initial2 = initial.reshape(mul(*initial.shape))
    # initial2[: int(load*initial_size**2)] = 1
    # np.random.shuffle(initial2)
    # initial = initial2.reshape(initial.shape)
    return initial


def rmc(image: np.array, T_MAX, N, initial: np.array = None):
    # simulated_image, t_max, t_min = hiprmc.temp_tuning(image, T_MAX, N, initial = initial)
    # T = t_max
    if initial is None:
        initial = hiprmc.random_initial(image)

    simulated_image = initial.copy()
    t_min = 0.0001
    iterations = 1000
    t_step = np.exp((np.log(t_min) - np.log(T_MAX)) / iterations)

    F_image = hiprmc.fourier_transform(image)
    F_old = hiprmc.fourier_transform(simulated_image)
    chi_old = hiprmc.chi_square(F_old, F_image)

    T = T_MAX

    accept_rate, temperature, error, iteration = [], [], [], []
    move_distance = int(N / 2)

    for t in progressbar.progressbar(range(0, iterations)):
        move_count = 0.0
        acceptance = 0.0
        T = T * t_step
        for iter in range(0, np.count_nonzero(simulated_image == 1)):
            move_count += 1
            x_old = np.random.randint(0, image.shape[0])
            y_old = np.random.randint(0, image.shape[1])
            while simulated_image[x_old][y_old] != 1:
                x_old = np.random.randint(0, image.shape[0])
                y_old = np.random.randint(0, image.shape[1])
            x_new = np.random.randint(0, image.shape[0])
            y_new = np.random.randint(0, image.shape[1])
            while np.sqrt((y_new - y_old) ** 2 + (x_new - x_old) ** 2) > move_distance:
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
            F_new = F_old - hiprmc.fourier_transform(old_array) + hiprmc.fourier_transform(new_array)
            chi_new = hiprmc.chi_square(F_new, F_image)
            delta_chi = chi_new - chi_old
            acceptance, chi_old = hiprmc.Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
                                                    simulated_image, T, acceptance, chi_old, chi_new, T_MAX, iter, N,
                                                    t_step)
            F_old = hiprmc.fourier_transform(simulated_image)

        acceptance_rate = acceptance / move_count
        accept_rate.append(acceptance_rate)
        temperature.append(T)
        error.append(chi_old)
        iteration.append(t)

        # if acceptance_rate > 0.5:
        #     move_distance = move_distance - 1
        # if move_distance <= 0:
        #     move_distance = 1
        # if acceptance_rate < 0.3:
        #     move_distance = move_distance + 1
        # if move_distance > N/2:
        #     move_distance = N/2

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(temperature, accept_rate, 'bo')
    plt.ylim([0, 1])
    plt.ylabel('Acceptance Rate')
    plt.xlabel('Temperature')
    plt.subplot(1, 2, 2)
    plt.plot(iteration, error, 'k-')
    plt.ylabel('Chi-Squared Error')
    plt.xlabel('Monte Carlo Iteration')
    plt.tight_layout()
    plt.show()

    return simulated_image
