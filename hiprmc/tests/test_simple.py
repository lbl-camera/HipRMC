import matplotlib.pyplot as plt
import numpy as np
import hiprmc
from operator import mul
from skimage.draw import polygon, circle
from PIL import Image

def test_simple():
    N = 20  # size of image (NxN)
    initial_image = np.zeros((N, N))
    iterations = 2000

    # This bit creates a disc to try and model via RMC
    # for i in range(0, N):
    #      for j in range(0, N):
    #          r = np.sqrt((i - int(N / 2)) ** 2 + (j - int(N / 2)) ** 2)
    #          if r >= 60 and r <= 61:
    #              initial_image[i, j] = 1
    #          else:
    #             initial_image[i, j] = 0

    # This creates a circle to model via RMC
    rr, cc = circle(int(N/4), int(N/4), int(N/8))
    initial_image[rr, cc] = 1

    # This creates a square
    #initial_image[int(2*N/6):int(4*N/6), int(1*N/6):int(5*N/6)] = 0

    f_image = initial_image

    # Image upload
    #f_image = np.load('/home/rp/Downloads/cool_40p0C_ILC1_insitu_40pt0C_2m.edf.npy')
    #f_mask = np.load('/home/nathan/shuai_processed_data/shuai_data_mask.edf.npy')
    #N = f_image.shape[0]
    #f_image[f_image==np.nan]=0
    #plt.imshow(np.log10(f_image), origin='lower', cmap='Greys')
    #plt.show()

    # set load of particles
    load = 0.4
    #load = np.count_nonzero(initial_image == 1)/N/N

    #random particles
    #initial_image = np.random.choice([0, 1], size=(N,N), p=[1.0-load, load])

    T = f_image.shape[0]  # temperature is on the same order as the image
    T_MAX = T

    #random_start = hiprmc.random_initial(f_image)
    initial = np.zeros_like(f_image)
    initial2 = initial.reshape(mul(*f_image.shape))
    initial2[: int(load*mul(*f_image.shape))] = 1
    np.random.shuffle(initial2)
    random_start = initial2.reshape(f_image.shape)
    #f_image = np.fft.fft2(initial_image)
    simulated_image = hiprmc.rmc(f_image, N, T_MAX, iterations)

    f = plt.figure()
    f.add_subplot(2, 2, 1)
    plt.imshow(f_image, cmap='Greys', origin="lower")
    plt.axis('off')
    plt.title("original image")
    #f.add_subplot(1, 5, 2)
    #plt.axis('off')
    #lt.title("Initial Intensity Plot")
    f.add_subplot(2, 2, 2)
    plt.imshow(np.log10(abs(hiprmc.fourier_transform(f_image))**2), origin="lower", cmap='Greys')
    #plt.imshow(mask, cmap='Greys',origin='lower', alpha=0.5)
    plt.axis('off')
    plt.title("Initial (Cropped) Intensity Plot")
    f.add_subplot(2, 2, 3)
    plt.imshow(simulated_image, cmap="Greys", origin="lower")
    plt.title("Model")
    plt.axis('off')
    f.add_subplot(2, 2, 4)
    plt.imshow(np.log10(abs(hiprmc.fourier_transform(simulated_image))**2), origin='lower', cmap='Greys')
    plt.axis('off')
    plt.title("Model Intensity Plot")
    plt.show()


if __name__ == '__main__':
    test_simple()
