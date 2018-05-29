import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import hiprmc
from skimage.draw import polygon

def test_simple():
    N = 25  # size of image (NxN)
    disc = np.zeros((N, N))
    T = 1000
    # This bit creates a disc to try and model via RMC
    for i in range(0, N):
        for j in range(0, N):
            r = (i - N / 2) ** 2 + (j - N / 2) ** 2
            if r < (N / 2 - 2) ** 2 and r > (N / 2 - 4) ** 2:
                disc[i, j] = 1
            else:
                disc[i, j] = 0

    # r = np.array([1, 12, 18, 1])
    # c = np.array([1, 17, 14, 1])
    # rr, cc = polygon(r, c)
    # disc[rr, cc] = 1

    # picture = Image.open('saxs10_128.tiff')
    # disc = misc.imread('saxs10_128.tiff')
    # print(disc)

    initial = hiprmc.random_initial(disc)
    simulated_image = hiprmc.rmc(disc, T, initial=initial)

    f = plt.figure()
    f.add_subplot(1, 5, 1)
    plt.imshow(disc, cmap='Greys', origin="lower")
    plt.axis('off')
    plt.title("original image")
    f.add_subplot(1, 5, 2)
    plt.imshow(np.log10(abs(hiprmc.fourier_transform(disc))), origin="lower")
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
