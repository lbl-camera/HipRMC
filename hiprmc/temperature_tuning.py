import progressbar
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import hiprmc


def temp_tuning(image: np.array, T_MAX, N, norm):
    #if initial is None:
    initial = hiprmc.random_initial(image)
    T = T_MAX
    # T = 0
    simulated_image = initial.copy()
    original = simulated_image
    F_image = hiprmc.fourier_transform(image)
    F_old = hiprmc.fourier_transform(simulated_image)
    i_image = hiprmc.abs2(f_image)
    i_simulation = hiprmc.abs2(f_old)
    norm = np.linalg.norm(f_image) ** 2
    chi_old = hiprmc.chi_square(F_old, F_image)
    error, count, accept_rate, temperature = [], [], [], []
    t_step = T_MAX / 100.0
    for t in progressbar.progressbar(range(0, 100)):
        move_count = 0.0
        acceptance = 0.0
        T = T - t_step
        for iter in range(0, np.count_nonzero(simulated_image == 1)):
            move_count += 1
            x_old = np.random.randint(0, image.shape[0])
            y_old = np.random.randint(0, image.shape[1])
            while simulated_image[x_old][y_old] != 1:
                x_old = np.random.randint(0, image.shape[0])
                y_old = np.random.randint(0, image.shape[1])
            x_new = np.random.randint(0, image.shape[0])
            y_new = np.random.randint(0, image.shape[1])
            while np.sqrt((y_new - y_old) ** 2 + (x_new - x_old) ** 2) > 5:
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
                                                    simulated_image, T, acceptance, chi_old, chi_new, T_MAX, iter, N)
            F_old = hiprmc.fourier_transform(simulated_image)
        error.append(chi_old)
        count.append(t)
        accept_rate.append(acceptance / move_count)
        temperature.append(T)

    # def gaussian(x, a, b):
    #     return a * (1 - np.exp(-(x / b) ** 2))
    def sigmoid(x, a, b):
        return 1.0 / (1.0 + np.exp(-a * (x - b)))

    fit_params, fit_params_error = optimize.curve_fit(sigmoid, temperature, accept_rate, p0=[0.5, T_MAX / 2])

    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # plt.plot(temperature, accept_rate, 'bo')
    # plt.plot(temperature, sigmoid(temperature, *fit_params), 'r-')
    # plt.ylabel('Acceptance Rate')
    # plt.ylim([0, 1])
    # plt.xlabel('Temperature')
    # f.add_subplot(1, 2, 2)
    # plt.plot(count, error, 'k-')
    # plt.ylabel('Chi-Squared')
    # plt.xlabel('Monte Carlo Iteration')
    # plt.tight_layout()
    # plt.show()

    # temp_tstar = abs(fit_params[1]) * np.sqrt(-np.log(1 - 0.333333 / fit_params[0]))
    temp_tstar = fit_params[1] - np.log(2.3333) / fit_params[0]
    t_min = max(0.001, temp_tstar - 0.2)
    t_max = min(T_MAX, temp_tstar + 0.2)
    t_step = (t_max - t_min) / 1000
    T = t_max
    simulated_image = original
    print(t_min, t_max)
    temperature2, accept_rate2 = [], []
    for t in progressbar.progressbar(range(0, 1000)):
        move_count = 0.0
        acceptance = 0.0
        for iter in range(0, np.count_nonzero(simulated_image == 1)):
            move_count += 1
            x_old = np.random.randint(0, image.shape[0])
            y_old = np.random.randint(0, image.shape[1])
            while simulated_image[x_old][y_old] != 1:
                x_old = np.random.randint(0, image.shape[0])
                y_old = np.random.randint(0, image.shape[1])
            x_new = np.random.randint(0, image.shape[0])
            y_new = np.random.randint(0, image.shape[1])
            while np.sqrt((y_new - y_old) ** 2 + (x_new - x_old) ** 2) > 15:
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
                                                    simulated_image, T, acceptance, chi_old, chi_new, T_MAX, iter, N)
            F_old = hiprmc.fourier_transform(simulated_image)

        accept_rate2.append(acceptance / move_count)
        temperature2.append(T)
        T = T - t_step

    fit_params, fit_params_error = optimize.curve_fit(sigmoid, temperature2, accept_rate2, p0=[0.5, temp_tstar])

    # temp_tstar = abs(fit_params[1]) * np.sqrt(-np.log(1 - 0.333333 / fit_params[0]))
    temp_tstar = fit_params[1] - np.log(2) / fit_params[0]
    t_min = min(0.001, temp_tstar - 0.2)
    t_max = min(T_MAX, temp_tstar + 0.2)
    #
    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # plt.plot(temperature2, accept_rate2, 'bo')
    # plt.plot(temperature2, sigmoid(temperature2, *fit_params), 'r-')
    # plt.ylabel('Acceptance Rate')
    # plt.ylim([0,1])
    # plt.xlabel('Temperature')
    # f.add_subplot(1, 2, 2)
    # plt.plot(count, error, 'k-')
    # plt.ylabel('Chi-Squared')
    # plt.xlabel('Monte Carlo Iteration')
    # plt.tight_layout()
    # plt.show()
    #
    # simulated_image = original

    # f = plt.figure()
    # f.add_subplot(1, 5, 1)
    # plt.axis('off')
    # plt.title("original image")
    # f.add_subplot(1, 5, 2)
    # plt.axis('off')
    # plt.title("FFT of Image")
    # f.add_subplot(1, 5, 3)
    # plt.imshow(initial, cmap="Greys", origin="lower")
    # plt.title("random start")
    # plt.axis('off')
    # f.add_subplot(1, 5, 4)
    # plt.imshow(simulated_image, cmap="Greys", origin="lower")
    # plt.title("updated")
    # plt.axis('off')
    # f.add_subplot(1, 5, 5)
    # plt.imshow(np.log10(abs(hiprmc.fourier_transform(simulated_image))), origin='lower')
    # plt.axis('off')
    # plt.title("FFT of final state")
    # plt.show()
    return simulated_image, t_max, t_min
