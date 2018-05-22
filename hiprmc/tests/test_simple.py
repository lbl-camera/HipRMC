import matplotlib.pyplot as plt
import numpy as np
import hiprmc

def test_simple():
    N = 25  # size of image (NxN)
    disc, rand_start, DFT = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
    T = 1.0

    # This bit creates a disc to try and model via RMC
    for i in range(0, N):
        for j in range(0, N):
            r = (i - N / 2) ** 2 + (j - N / 2) ** 2
            if r < (N / 2 - 2) ** 2 and r > (N / 2 - 4) ** 2:
                disc[i, j] = 1
            else:
                disc[i, j] = 0

    plt.imshow(disc, cmap='Greys', origin="lower")
    plt.axis('off')
    plt.show()

    initial = hiprmc.random_initial(disc)
    simulated_image = hiprmc.rmc(disc, T, initial=initial)

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(initial, cmap="Greys", origin="lower")
    plt.title("original")
    f.add_subplot(1, 2, 2)
    plt.imshow(simulated_image, cmap="Greys", origin="lower")
    plt.title("updated")
    plt.show()

    # plt.imshow(disc, cmap='Greys', origin="lower")
    # plt.axis('off')
    # plt.show()
    # plt.imshow(np.log10(abs(fourier_transform(disc))), origin='lower', cmap='Greys')
    # plt.axis('off')
    # plt.show()
