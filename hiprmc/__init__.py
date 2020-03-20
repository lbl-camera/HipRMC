import numpy as np
from scipy import fftpack
from scipy.linalg import dft
from operator import mul
from hiprmc.temperature_tuning import temp_tuning
import progressbar
from functools import lru_cache
import numba
import matplotlib.pyplot as plt
from matplotlib import animation

def fourier_transform(m):
    return fftpack.fft2(m)

@lru_cache(1)
def memoized_dft(N):
    return dft(N)

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

@lru_cache(1)
def q_scaling(shape):
    return np.fromfunction(lambda i, j: ((i-shape[0]/2)**2+(i-shape[1])**2/2)**-.5, shape)

@lru_cache(1)
def lin_scaling(shape):
    return np.fromfunction(lambda i, j: 1 - 1/50. * ((i-shape[0]/2)**2+(i-shape[1])**2/2)**.5, shape)

@numba.jit(nopython=True)
def chi_square(i_simulation, i_image, norm):
    return np.sum(abs2(i_image - i_simulation)) / norm

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

def random_initial(image:np.array, load):
    initial = np.zeros_like(image)
    initial2 = initial.reshape(mul(*image.shape))
    initial2[: int(load*image.size)] = 1
    np.random.shuffle(initial2)
    initial = initial2.reshape(image.shape)
    return initial


def rmc(image: np.array, T_MAX, iterations, load, movie=False):

    ##########################################################
    ##### Temp Tuning:  Uncomment for Tuning ##################
    ##########################################################
    print('Performing Temperature Tuning')
    simulated_image = random_initial(image, load)
    t_max, t_min = temp_tuning(image, simulated_image, T_MAX)
    T = t_max
    print('Completed Temperature Tuning')
    print(t_max,t_min)
    ##########################################################
    ##### Temp Tuning:  comment for Tuning ###################
    ##########################################################
    #simulated_image = random_initial(image, load)
    #t_min = 0.00001
    #T = T_MAX
    ##########################################################
    ######### End Temp Tuning Indents ########################
    ##########################################################

    iterations = iterations
    t_step = np.exp((np.log(t_min) - np.log(T)) / iterations)   # exponential cooling rate
    #t_step = (T_MAX - t_min)/iterations                             # linear cooling rate
    #if movie:
    #    import cv2
    #    from cv2 import VideoWriter, VideoWriter_fourcc
    #    fourcc = VideoWriter_fourcc(*'XVID')
    #    video = VideoWriter('./test.avi', fourcc, float(60), image.shape)

    f_old = fourier_transform(simulated_image)
    i_image = image
    i_simulation = abs2(f_old)
    norm = np.linalg.norm(image)**2   ########change what you take the norm of, f_image or i_image
    chi_old = chi_square(i_simulation, i_image, norm)

    accept_rate, temperature, error, iteration = [], [], [], []
    move_distance = int(image.shape[0] / 2)

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
            old_point = new_point = simulated_image[x_old][y_old]
            while new_point == old_point:
                x_new, y_new = random_from_circle(move_distance, x_old, y_old)
                x_new, y_new = periodic(image, x_new, y_new)
                new_point = simulated_image[x_new][y_new]
            f_new = f_old + DFT_Matrix(x_old, y_old, x_new, y_new, image.shape[0])
            i_simulation = abs2(f_new)
            chi_new = chi_square(i_simulation, i_image, norm)
            delta_chi = chi_new - chi_old
            acceptance, chi_old = Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
                                                    simulated_image, T, acceptance, chi_old, chi_new, T_MAX, iter, image.shape[0],
                                                    t_step)
            f_old = fourier_transform(simulated_image)

        particle_list = list(zip(*np.nonzero(simulated_image)))

        #if movie:
        #    frame = simulated_image.astype(np.uint8)*255
        #    video.write(np.dstack([frame, frame, frame]))



        acceptance_rate = acceptance / move_count
        accept_rate.append(acceptance_rate)
        temperature.append(T)
        error.append(chi_old)
        iteration.append(t)
    print(np.mean(accept_rate))
    plt.figure(3)
    plt.plot((temperature),accept_rate,'ro')
    plt.title('Acceptance Rate')
    plt.ylabel('Acceptance Rate')
    plt.xlabel('Temperature')
    plt.xscale('log')
    plt.show()


    #if movie:
    #    video.release()

    return simulated_image