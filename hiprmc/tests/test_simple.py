import matplotlib.pyplot as plt
import numpy as np
import hiprmc
from skimage.draw import polygon, circle
from PIL import Image

def test_simple():
    N = 100  # size of image (NxN)
    initial_image = np.zeros((N, N))

    # This bit creates a disc to try and model via RMC
    # for i in range(0, N):
    #     for j in range(0, N):
    #         r = np.sqrt((i - int(N / 2)) ** 2 + (j - int(N / 2)) ** 2)
    #         if r >= 5 and r <= 7:
    #             initial_image[i, j] = 1
    #         else:
    #             initial_image[i, j] = 0

    # This creates a circle to model via RMC
    rr, cc = circle(int(N / 2), int(N / 2), int(N / 4))
    initial_image[rr, cc] = 1

    # Image upload
    # img = Image.open('test_cold.tif')
    # initial_image = np.array(img)
    # initial_image[np.isnan(initial_image)] = 1
    # N = initial_image.shape[0]

    # set load of particles
    # load = 0.05 * N * N

    # declare whether data is simulated or real
    mode = 'simulated_data'

    T = N  # temperature is on the same order as the image
    T_MAX = T

    initial = hiprmc.random_initial(initial_image)
    simulated_image = hiprmc.rmc(initial_image, T_MAX, N, mode, initial=initial)

    f = plt.figure()
    f.add_subplot(1, 5, 1)
    plt.imshow(initial_image, cmap='Greys', origin="lower")
    plt.axis('off')
    plt.title("original image")
    # f.add_subplot(1, 5, 2)
    # plt.imshow(np.log10(abs(hiprmc.fourier_transform(initial_image))), origin="lower")
    # plt.axis('off')
    #plt.title("FFT of Image")
    f.add_subplot(1, 5, 3)
    plt.imshow(initial, cmap="Greys", origin="lower")
    plt.title("random start")
    plt.axis('off')
    f.add_subplot(1, 5, 4)
    plt.imshow(simulated_image, cmap="Greys", origin="lower")
    plt.title("updated")
    plt.axis('off')
    f.add_subplot(1, 5, 5)
    plt.imshow(np.log10(abs(hiprmc.fourier_transform(simulated_image)) ** 2), origin='lower')
    plt.axis('off')
    plt.title("FFT of final state")
    plt.show()


if __name__ == '__main__':
    test_simple()
