import numpy as np
from scipy.linalg import dft
from operator import mul
from .temperature_tuning import temp_tuning
import progressbar
from numpy.fft import fft2 as fourier_transform
from functools import lru_cache
import numba

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

@numba.jit(nopython=True)
def chi_square(i_simulation, i_image, norm, mask):
    return np.sum(abs2(i_simulation - i_image) / norm * mask)

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
            # temperature = T / (T_MAX - T)
            b = np.exp(-delta_chi / T)  # * np.count_nonzero(simulated_image == 1) * N ** 2 / (1 + t_step * iter))
            if b > np.random.rand():
                simulated_image[x_old][y_old] = new_point
                simulated_image[x_new][y_new] = old_point
                acceptance += 1.0
                chi_old = chi_new
            else:
                simulated_image[x_old][y_old] = old_point
                simulated_image[x_new][y_new] = new_point
    return acceptance, chi_old


def random_initial(image: np.array):
    initial = np.zeros_like(image)
    initial2 = initial.reshape(mul(*image.shape))
    initial2[: np.count_nonzero(image == 1)] = 1
    np.random.shuffle(initial2)
    initial = initial2.reshape(image.shape)
    return initial


def random_initial_crop(image: np.array, load):
    initial = np.zeros_like(image)
    initial2 = initial.reshape(mul(*image.shape))
    initial2[: int(load * mul(*image.shape))] = 1
    np.random.shuffle(initial2)
    initial = initial2.reshape(image.shape)
    return initial


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


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def rmc(image: np.array, mask: np.array, T_MAX, iterations, load, random_start: np.array = None):
    # uncomment the below two lines if using temperature tuning
    # simulated_image, t_max, t_min = hiprmc.temp_tuning(image, T_MAX, N, initial = initial)
    # T = t_max

    mask=mask.astype(np.bool_)

    # comment the three lines below if using using temperature tuning
    t_min = 0.0001
    T = T_MAX

    iterations = iterations
    t_step = np.exp((np.log(t_min) - np.log(T_MAX)) / iterations)

    accept_rate, temperature, error, iteration = [], [], [], []
    N = 80
    initial_crop = crop_center(image, N, N)
    mask = np.logical_not(mask)
    mask = crop_center(mask, N, N)

    random_start_small = random_initial_crop(initial_crop, load)
    F_old = fourier_transform(random_start_small)

    if random_start is None:
        random_start = random_initial(image)

    simulated_image = np.real(random_start_small)

    i_simulation = abs2(F_old)
    i_image = initial_crop
    norm = np.linalg.norm(i_image) ** 2

    chi_old = chi_square(i_simulation, i_image, norm, mask)

    particle_list = list(zip(*np.nonzero(simulated_image)))

    for t in progressbar.progressbar(range(0, iterations)):
        move_count = 0.0
        acceptance = 0.0
        T = T * t_step
        for particle_number in range(0, len(particle_list)):
            move_count += 1
            x_old = particle_list[particle_number][0]
            y_old = particle_list[particle_number][1]
            x_new = np.random.randint(0, initial_crop.shape[0])
            y_new = np.random.randint(0, initial_crop.shape[1])
            move_distance = int(N / 2)
            old_point = new_point = simulated_image[x_old][y_old]

            while new_point == old_point:
                x_new, y_new = random_from_circle(move_distance, x_old, y_old)
                x_new, y_new = periodic(initial_crop, x_new, y_new)
                new_point = simulated_image[x_new][y_new]
            simulated_image[x_old][y_old] = new_point
            simulated_image[x_new][y_new] = old_point
            F_new = F_old + DFT_Matrix(x_old, y_old, x_new, y_new, initial_crop.shape[0])
            i_simulation = abs2(F_new)
            chi_new = chi_square(i_simulation, i_image, norm, mask)
            delta_chi = chi_new - chi_old
            acceptance, chi_old = Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
                                                    simulated_image, T, acceptance, chi_old, chi_new, T_MAX,
                                                    particle_number, N,
                                                    t_step)
            F_old = F_new

        particle_list = list(zip(*np.nonzero(simulated_image)))

        acceptance_rate = acceptance / move_count
        accept_rate.append(acceptance_rate)
        temperature.append(T)
        error.append(chi_old)
        iteration.append(t)

    return simulated_image, initial_crop, mask
