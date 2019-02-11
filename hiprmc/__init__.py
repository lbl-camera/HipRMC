import numpy as np
from scipy.linalg import dft
from operator import mul
from hiprmc.temperature_tuning import temp_tuning
import hiprmc
import progressbar
import matplotlib.pyplot as plt
import cupy as cp
import arrayfire as af
import multiprocessing

af.set_backend('cuda')

def fourier_transform(m):
    m_fft = np.fft.fft2(m)
    return m_fft

def chi_square(i_simulation, i_image, norm, mask):
    return np.sum(np.multiply(np.square(abs(i_simulation - i_image)) / norm, mask))

def DFT_Matrix(x_old, y_old, x_new, y_new, N):
    m = dft(N)
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
            #temperature = T / (T_MAX - T)
            b = np.exp(-delta_chi / T)# * np.count_nonzero(simulated_image == 1) * N ** 2 / (1 + t_step * iter))
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

def random_initial_crop(image:np.array, load):
    initial = np.zeros_like(image)
    initial2 = initial.reshape(mul(*image.shape))
    initial2[: int(load*mul(*image.shape))] = 1
    np.random.shuffle(initial2)
    initial = initial2.reshape(image.shape)
    return initial

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def rmc(image: np.array, mask: np.array, T_MAX, iterations, load, random_start: np.array = None):
    # uncomment the below two lines if using temperature tuning
    # simulated_image, t_max, t_min = hiprmc.temp_tuning(image, T_MAX, N, initial = initial)
    # T = t_max

    # comment the three lines below if using using temperature tuning
    t_min = 0.0001
    T = T_MAX

    iterations = iterations
    t_step = np.exp((np.log(t_min) - np.log(T_MAX)) / iterations)

    accept_rate, temperature, error, iteration = [], [], [], []
    N = 80
    initial_crop = crop_center(image,N,N)
    mask = 1 - mask
    mask = crop_center(mask,N,N)
    #initial_crop[mask==0]=0
    plt.imshow(np.log10(initial_crop), cmap='Greys', alpha=0.9)
    plt.title('Scattering Pattern to be Simulated')
    plt.show()

    random_start_small = random_initial_crop(initial_crop,load)
    F_old = hiprmc.fourier_transform(random_start_small)

    if random_start is None:
        random_start = hiprmc.random_initial(image)

    simulated_image = np.real(random_start_small)

    i_simulation = abs(F_old)**2
    i_image = initial_crop
    norm = np.linalg.norm(i_image)**2

    chi_old = hiprmc.chi_square(i_simulation, i_image, norm, mask)

    particle_list = []
    for i in range(0, initial_crop.shape[0]):
        for j in range(0,initial_crop.shape[1]):
            if simulated_image[i][j] == 1:
                particle_list.append([i,j])

    for t in progressbar.progressbar(range(0, iterations)):
        move_count = 0.0
        acceptance = 0.0
        T = T * t_step
        for particle_number in range(0, np.count_nonzero(simulated_image == 1)):
            move_count += 1
            x_old = particle_list[particle_number][0]
            y_old = particle_list[particle_number][1]
            x_new = np.random.randint(0, initial_crop.shape[0])
            y_new = np.random.randint(0, initial_crop.shape[1])
            move_distance = int(N / 2)
            while np.sqrt((y_new - y_old) ** 2 + (x_new - x_old) ** 2) > move_distance:
                x_new = np.random.randint(0, initial_crop.shape[0])
                y_new = np.random.randint(0, initial_crop.shape[1])
            old_point = simulated_image[x_old][y_old]
            new_point = simulated_image[x_new][y_new]
            while new_point == old_point:
                x_new = np.random.randint(0, initial_crop.shape[0])
                y_new = np.random.randint(0, initial_crop.shape[1])
                new_point = simulated_image[x_new][y_new]
            simulated_image[x_old][y_old] = new_point
            simulated_image[x_new][y_new] = old_point
            F_new = F_old + DFT_Matrix(x_old, y_old, x_new, y_new, initial_crop.shape[0])
            i_simulation = abs(F_new)**2
            chi_new = hiprmc.chi_square(i_simulation, i_image, norm, mask)
            delta_chi = chi_new - chi_old
            acceptance, chi_old = hiprmc.Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
                                                    simulated_image, T, acceptance, chi_old, chi_new, T_MAX, particle_number, N,
                                                    t_step)
            F_old = hiprmc.fourier_transform(simulated_image)

        particle_list = []
        for i in range(0, initial_crop.shape[0]):
            for j in range(0, initial_crop.shape[1]):
                if simulated_image[i][j] == 1:
                    particle_list.append([i, j])

        acceptance_rate = acceptance / move_count
        accept_rate.append(acceptance_rate)
        temperature.append(T)
        error.append(chi_old)
        iteration.append(t)

    return simulated_image, initial_crop, mask
