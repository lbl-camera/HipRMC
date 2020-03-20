import progressbar
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import hiprmc


def temp_tuning(image: np.array, simulated_image: np.array,  T_MAX):

    p = 0.5
    tuning_iterations = 20
    t_step = np.exp((np.log(0.000000001) - np.log(T_MAX)) / tuning_iterations)
    #t_step = T_MAX / 1000.0
    T = T_MAX

    f_old = hiprmc.fourier_transform(simulated_image)
    i_image = image
    i_simulation = hiprmc.abs2(f_old)
    norm = np.linalg.norm(image)**2   ########change what you take the norm of, f_image or i_image
    chi_old = hiprmc.chi_square(i_simulation, i_image, norm)

    accept_rate, temperature, error, iteration = [], [], [], []
    move_distance = int(image.shape[0] / 2)

    particle_list = list(zip(*np.nonzero(simulated_image)))

    for t in progressbar.progressbar(range(0, tuning_iterations)):
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
                x_new, y_new = hiprmc.random_from_circle(move_distance, x_old, y_old)
                x_new, y_new = hiprmc.periodic(image, x_new, y_new)
                new_point = simulated_image[x_new][y_new]
            f_new = f_old + hiprmc.DFT_Matrix(x_old, y_old, x_new, y_new, image.shape[0])
            i_simulation = hiprmc.abs2(f_new)
            chi_new = hiprmc.chi_square(i_simulation, i_image, norm)
            delta_chi = chi_new - chi_old
            acceptance, chi_old = hiprmc.Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
                                                    simulated_image, T, acceptance, chi_old, chi_new, T_MAX, iter,
                                                    image.shape[0],
                                                    t_step)
            f_old = hiprmc.fourier_transform(simulated_image)

        particle_list = list(zip(*np.nonzero(simulated_image)))

        acceptance_rate = acceptance / move_count
        accept_rate.append(acceptance_rate)
        temperature.append(T)
        error.append(chi_old)
        iteration.append(t)
    plt.figure(5)
    plt.plot(np.log(temperature), accept_rate, 'ro')
    plt.title('Acceptance Rate')
    plt.ylabel('Acceptance Rate')
    plt.xlabel('ln(Temperature)')
    #plt.xscale('log')
    plt.show()

    def sigmoid(x, a, b, c, L):
       return L / (1.0 + np.exp(-a * (x - b))) + c

    p0 = [max(accept_rate), np.median(temperature), 1, min(accept_rate)]
    fit_params, fit_params_error = optimize.curve_fit(sigmoid, np.log(temperature), accept_rate, p0, maxfev=6000, method='dogbox')

    temp_tstar = fit_params[1] - (fit_params[3]/fit_params[0])*np.log((1.0/p)-1.0)
    print(temp_tstar)

    plt.figure(4)
    plt.plot(np.log(temperature), accept_rate, 'bo')
    plt.plot(np.log(temperature), sigmoid(np.log(temperature), *fit_params), 'r-')
    plt.ylabel('Acceptance Rate')
    plt.xlabel('Temperature')
    #plt.xscale('log')
    plt.show()
    # g.add_subplot(1, 2, 2)
    # plt.plot(count, error, 'k-')
    # plt.ylabel('Chi-Squared')
    # plt.xlabel('Monte Carlo Iteration')
    # plt.tight_layout()
    # plt.show()

    t_min = max(np.log(0.0001), temp_tstar-2.0)
    t_max = min(np.log(T_MAX), temp_tstar+2.0)
    print(t_min, t_max)

   ############## TEMP RANGE 1 ###################


    T = np.exp(t_max)
    T_min = np.exp(t_min)

    tuning_iterations = 200

    t_step = np.exp((np.log(T_min) - np.log(T)) / tuning_iterations)

    f_old = hiprmc.fourier_transform(simulated_image)
    i_image = image
    i_simulation = hiprmc.abs2(f_old)
    norm = np.linalg.norm(image) ** 2  ########change what you take the norm of, f_image or i_image
    chi_old = hiprmc.chi_square(i_simulation, i_image, norm)

    accept_rate, temperature, error, iteration = [], [], [], []
    move_distance = int(image.shape[0] / 2)

    particle_list = list(zip(*np.nonzero(simulated_image)))

    for t in progressbar.progressbar(range(0, tuning_iterations)):
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
                x_new, y_new = hiprmc.random_from_circle(move_distance, x_old, y_old)
                x_new, y_new = hiprmc.periodic(image, x_new, y_new)
                new_point = simulated_image[x_new][y_new]
            f_new = f_old + hiprmc.DFT_Matrix(x_old, y_old, x_new, y_new, image.shape[0])
            i_simulation = hiprmc.abs2(f_new)
            chi_new = hiprmc.chi_square(i_simulation, i_image, norm)
            delta_chi = chi_new - chi_old
            acceptance, chi_old = hiprmc.Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
                                                    simulated_image, T, acceptance, chi_old, chi_new, T_MAX, iter,
                                                    image.shape[0],
                                                    t_step)
            f_old = hiprmc.fourier_transform(simulated_image)

        particle_list = list(zip(*np.nonzero(simulated_image)))

        acceptance_rate = acceptance / move_count
        accept_rate.append(acceptance_rate)
        temperature.append(T)
        error.append(chi_old)
        iteration.append(t)

    plt.figure(6)
    plt.plot(np.log(temperature), accept_rate, 'ro')
    plt.title('Acceptance Rate')
    plt.ylabel('Acceptance Rate')
    plt.xlabel('ln(Temperature)')
    #plt.xscale('log')
    plt.show()

    p0 = [max(accept_rate), np.median(temperature), 1, min(accept_rate)]
    fit_params, fit_params_error = optimize.curve_fit(sigmoid, np.log(temperature), accept_rate, p0, maxfev=6000, method='dogbox')
    temp_tstar = fit_params[1] - (fit_params[3]/fit_params[0])*np.log((1.0/p)-1.0)
    print(temp_tstar)

    plt.figure(4)
    plt.plot(np.log(temperature), accept_rate, 'ro')
    plt.plot(np.log(temperature), sigmoid(np.log(temperature), *fit_params), 'b-')
    plt.ylabel('Acceptance Rate')
    plt.xlabel('Temperature')
    plt.show()

    t_min = max(t_min, temp_tstar-2.0)
    t_max = min(t_max, temp_tstar+2.0)

    ############## TEMP RANGE 2 ###################


    # T = np.exp(t_max)
    # t_min = np.exp(t_min)
    #
    # tuning_iterations = 1000
    #
    # t_step = np.exp((np.log(t_min) - np.log(T)) / tuning_iterations)
    #
    # f_old = hiprmc.fourier_transform(simulated_image)
    # i_image = image
    # i_simulation = hiprmc.abs2(f_old)
    # norm = np.linalg.norm(image) ** 2  ########change what you take the norm of, f_image or i_image
    # chi_old = hiprmc.chi_square(i_simulation, i_image, norm)
    #
    # accept_rate, temperature, error, iteration = [], [], [], []
    # move_distance = int(image.shape[0] / 2)
    #
    # particle_list = list(zip(*np.nonzero(simulated_image)))
    #
    # for t in progressbar.progressbar(range(0, tuning_iterations)):
    #     move_count = 0.0
    #     acceptance = 0.0
    #     T = T * t_step
    #     for particle_number in range(0, len(particle_list)):
    #         move_count += 1
    #         x_old = particle_list[particle_number][0]
    #         y_old = particle_list[particle_number][1]
    #         x_new = np.random.randint(0, image.shape[0])
    #         y_new = np.random.randint(0, image.shape[1])
    #         old_point = new_point = simulated_image[x_old][y_old]
    #         while new_point == old_point:
    #             x_new, y_new = hiprmc.random_from_circle(move_distance, x_old, y_old)
    #             x_new, y_new = hiprmc.periodic(image, x_new, y_new)
    #             new_point = simulated_image[x_new][y_new]
    #         f_new = f_old + hiprmc.DFT_Matrix(x_old, y_old, x_new, y_new, image.shape[0])
    #         i_simulation = hiprmc.abs2(f_new)
    #         chi_new = hiprmc.chi_square(i_simulation, i_image, norm)
    #         delta_chi = chi_new - chi_old
    #         acceptance, chi_old = hiprmc.Metropolis(x_old, y_old, x_new, y_new, old_point, new_point, delta_chi,
    #                                                 simulated_image, T, acceptance, chi_old, chi_new, T_MAX, iter,
    #                                                 image.shape[0],
    #                                                 t_step)
    #         f_old = hiprmc.fourier_transform(simulated_image)
    #
    #     particle_list = list(zip(*np.nonzero(simulated_image)))
    #
    #     acceptance_rate = acceptance / move_count
    #     accept_rate.append(acceptance_rate)
    #     temperature.append(T)
    #     error.append(chi_old)
    #     iteration.append(t)
    #
    # plt.figure(6)
    # plt.plot(np.log(temperature), accept_rate, 'ro')
    # plt.title('Acceptance Rate')
    # plt.ylabel('Acceptance Rate')
    # plt.xlabel('ln(Temperature)')
    # #plt.xscale('log')
    # plt.show()
    #
    # p0 = [max(accept_rate), np.median(temperature), 1, min(accept_rate)]
    # fit_params, fit_params_error = optimize.curve_fit(sigmoid, np.log(temperature), accept_rate, p0, maxfev=6000, method='dogbox')
    #
    # tstar = fit_params[1] - (fit_params[3]/fit_params[0])*np.log((1.0/p)-1.0)
    #
    # plt.figure(4)
    # plt.plot(np.log(temperature), accept_rate, 'ro')
    # plt.plot(np.log(temperature), sigmoid(np.log(temperature), *fit_params), 'b-')
    # plt.ylabel('Acceptance Rate')
    # plt.xlabel('Temperature')
    # #plt.xscale('log')
    # plt.show()
    #
    # t_max = np.exp(tstar + 1.0)
    # t_min = np.exp(tstar - 1.0)
    print(t_max,t_min)
    return np.exp(t_max), np.exp(t_min)
