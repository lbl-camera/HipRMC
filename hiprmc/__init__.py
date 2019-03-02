import numpy as np
from scipy import fftpack
from scipy.linalg import dft
from operator import mul
from hiprmc.temperature_tuning import temp_tuning
import progressbar
from functools import lru_cache
import numba

def fourier_transform(m):
    return fftpack.fft2(m)

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

@numba.jit(nopython=True)
def chi_square(i_simulation, i_image, norm):
    return np.sum(abs2(i_image - i_simulation) / norm)

@lru_cache(1)
def memoized_dft(N):
    return dft(N)

def DFT_Matrix(x_old, y_old, x_new, y_new, N):
    m = memoized_dft(N)
    before_row, before_column = m[x_old, :], m[:, y_old]
    after_row, after_column = m[x_new, :], m[:, y_new]
    U_new = np.outer(after_row, after_column)
    U_old = np.outer(before_row, before_column)
    U = U_new - U_old
    return U

def Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi, simulated_image, T, acceptance, chi_old,
               chi_new, T_MAX, iter, N, t_step):
    if T > 0:
        if delta_chi < 0:
            simulated_image[x_old][y_old] = new_point
            simulated_image[x_new][y_new] = old_point
            acceptance += 1.0
            chi_old = chi_new
        else:
            #temperature  = T / (T_MAX - T)
            b = np.exp(-delta_chi / T)
            if b > np.random.rand():
                simulated_image[x_old][y_old] = new_point
                simulated_image[x_new][y_new] = old_point
                acceptance += 1.0
                chi_old = chi_new
    return acceptance, chi_old

def random_from_circle(radius, centerX, centerY):
    r = radius * np.sqrt(np.random.random())
    theta = np.random.random() * 2 * np.pi
    x = centerX + r * np.cos(theta)
    y = centerY + r * np.sin(theta)
    return int(x), int(y)


def periodic(img, x, y):
    x = x % img.shape[0]
    y = y % img.shape[1]
    return x, y

def random_initial(image:np.array):
    initial = np.zeros_like(image)
    initial2 = initial.reshape(mul(*image.shape))
    initial2[: np.count_nonzero(image == 1)] = 1
    np.random.shuffle(initial2)
    initial = initial2.reshape(image.shape)
    return initial


def rmc(image: np.array, T_MAX, N, iterations):
    # uncomment the below two lines if using temperature tuning
    # simulated_image, t_max, t_min = hiprmc.temp_tuning(image, T_MAX, N, initial = initial)
    # T = t_max

   # if initial is None:
    initial = random_initial(image)

    # comment the three lines below if using using temperature tuning
    simulated_image = initial.copy()
    t_min = 0.0001
    T = T_MAX

    iterations = iterations
    t_step = np.exp((np.log(t_min) - np.log(T_MAX)) / iterations)

    f_image = fourier_transform(image)
    f_old = fourier_transform(simulated_image)
    i_image = abs2(f_image)
    i_simulation = abs2(f_old)
    norm = np.linalg.norm(f_image) ** 2
    chi_old = chi_square(i_simulation, i_image, norm)

    accept_rate, temperature, error, iteration = [], [], [], []
    move_distance = int(N / 2)

    particle_list = list(zip(*np.nonzero(simulated_image)))

    for t in progressbar.progressbar(range(0, iterations)):
        move_count = 0.0
        acceptance = 0.0
        T = T * t_step
        for particle_number in range(0, len(particle_list)):
            move_count += 1
            x_old = particle_list[particle_number][0]
            y_old = particle_list[particle_number][1]
            x_new = np.random.randint(0, image.shape[0])
            y_new = np.random.randint(0, image.shape[1])
            move_distance = int(N / 2)
            old_point = new_point = simulated_image[x_old][y_old]
            while new_point == old_point:
                x_new, y_new = random_from_circle(move_distance, x_old, y_old)
                x_new, y_new = periodic(image, x_new, y_new)
                new_point = simulated_image[x_new][y_new]
            f_new = f_old + DFT_Matrix(x_old, y_old, x_new, y_new, N)
            i_simulation = abs2(f_new)
            chi_new = chi_square(i_simulation, i_image, norm)
            delta_chi = chi_new - chi_old
            acceptance, chi_old = Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
                                                    simulated_image, T, acceptance, chi_old, chi_new, T_MAX, iter, N,
                                                    t_step)
            f_old = fourier_transform(simulated_image)

        particle_list = list(zip(*np.nonzero(simulated_image)))

        acceptance_rate = acceptance / move_count
        accept_rate.append(acceptance_rate)
        temperature.append(T)
        error.append(chi_old)
        iteration.append(t)

    return simulated_image
