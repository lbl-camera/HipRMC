import matplotlib.pyplot as plt
import numpy as np
import hiprmc
from skimage.draw import polygon, circle

def test_simple():
    N = 100  # size of image (NxN)
    initial_image = np.zeros((N, N))
    T = N  # temperature is on the same order as the image
    T_MAX = T

    # This bit creates a disc to try and model via RMC
    # for i in range(0, N):
    #     for j in range(0, N):
    #         r = np.sqrt((i - int(N / 2)) ** 2 + (j - int(N / 2)) ** 2)
    #         if r >= 5 and r <= 7:
    #             initial_image[i, j] = 1
    #         else:
    #             initial_image[i, j] = 0

    # initial_image = np.zeros((N,N))
    # initial2 = initial_image.reshape(mul(*initial_image.shape))
    # initial2[: num_particles] = 1
    # np.random.shuffle(initial2)
    # initial_image = initial2.reshape(initial_image.shape)

    # r = np.array([1, 20, 20, 1, 1])
    # c = np.array([1, 1, 5, 5, 1])
    # rr, cc = polygon(r, c)
    rr, cc = circle(int(N / 2), int(N / 2), int(N / 4))
    initial_image[rr, cc] = 1

    # picture = Image.open('saxs10_128.tiff')
    # disc = misc.imread('saxs10_128.tiff')
    # print(disc)
    image_shape = N
    initial = hiprmc.random_initial(initial_image)
    simulated_image = hiprmc.rmc(initial_image, T_MAX, image_shape, initial=initial)

    f = plt.figure()
    f.add_subplot(1, 5, 1)
    plt.imshow(initial_image, cmap='Greys', origin="lower")
    plt.axis('off')
    plt.title("original image")
    f.add_subplot(1, 5, 2)
    plt.imshow(np.log10(abs(hiprmc.fourier_transform(initial_image))), origin="lower")
    plt.axis('off')
    plt.title("FFT of Image")
    f.add_subplot(1, 5, 3)
    plt.imshow(initial, cmap="Greys", origin="lower")
    plt.title("random start")
    plt.axis('off')
    f.add_subplot(1, 5, 4)
    plt.imshow(simulated_image, cmap="Greys", origin="lower")
    plt.title("updated")
    plt.axis('off')
    f.add_subplot(1, 5, 5)
    plt.imshow(np.log10(abs(hiprmc.fourier_transform(simulated_image))), origin='lower')
    plt.axis('off')
    plt.title("FFT of final state")
    plt.show()


if __name__ == '__main__':
    test_simple()
