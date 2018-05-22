import numpy as np
from scipy import fftpack
from operator import mul


def fourier_transform(m):
    return fftpack.fft2(m)

def chi_square(simulation, image):
    return np.sum((np.square(abs(simulation - image))/abs(image)))

def DFT_Matrix(x_old, y_old, x_new, y_new, simulated_image):
    U_new = np.outer(simulated_image[x_new, :], simulated_image[:, y_new])
    U_old = np.outer(simulated_image[x_old, :], simulated_image[:, y_old])
    U = U_new - U_old
    return U


def Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi, simulated_image, T, acceptance):
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
    return acceptance

def random_initial(image:np.array):
    noise = np.random.random(image.shape)
    initial = np.zeros_like(image)
    initial[noise > np.count_nonzero(image == 1) / mul(*image.shape)] = 0
    initial[noise < np.count_nonzero(image == 1) / mul(*image.shape)] = 1
    return initial

def rmc(image:np.array, T, initial:np.array=None):
    if initial is None:
        initial = random_initial(image)

    simulated_image = initial.copy()
    F_image = fourier_transform(initial)
    for t in range(0, 50):
        move_count = 0.0
        acceptance = 0.0
        for _ in range(mul(*image.shape)):
            print(_)
            move_count += 1
            F_old = fourier_transform(simulated_image)
            chi_old = chi_square(F_old, F_image)
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
            F_new = fourier_transform(simulated_image) + DFT_Matrix(x_old, y_old, x_new, y_new, simulated_image)
            chi_new = chi_square(F_new, F_image)
            delta_chi = chi_new - chi_old
            acceptance = Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi, simulated_image, T, acceptance)
        print("The Acceptance Rate is", acceptance / move_count)

    return simulated_image
