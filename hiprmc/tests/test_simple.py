import matplotlib.pyplot as plt
import numpy as np
import hiprmc
from operator import mul
from skimage.draw import polygon, circle
from PIL import Image

def test_simulated():


    #####################################################
    ########## Begin Test Images ########################
    #####################################################

    N = 50  # size of image (NxN)
    initial_image = np.zeros((N, N))
    iterations = 2000

    # This bit creates a disc to try and model via RMC
    # for i in range(0, N):
    #     for j in range(0, N):
    #         r = np.sqrt((i - int(N / 2)) ** 2 + (j - int(N / 2)) ** 2)
    #         if 10 <= r <= 11:
    #             initial_image[i, j] = 1
    #         else:
    #             initial_image[i, j] = 0

    # # This creates a circle to model via RMC
    rr, cc = circle(int(N/2), int(N/2), int(N/8))
    initial_image[rr, cc] = 1

    # This creates a square
    #initial_image[int(2*N/6):int(4*N/6), int(1*N/6):int(2*N/6)] = 1

    #f_image = initial_image

    num_particles = np.count_nonzero(initial_image == 1)
    load = num_particles/initial_image.size

    i_image = hiprmc.abs2(hiprmc.fourier_transform(initial_image))

    T = 1000.0  # i_image.shape[0]  # temperature is on the same order as the image

    simulated_image = hiprmc.rmc(i_image, T, iterations, load, movie=True)

    f = plt.figure(3)
    f.add_subplot(2, 2, 1)
    plt.imshow(initial_image, cmap='Greys', origin="lower")
    plt.axis('off')
    plt.title("original image")
    f.add_subplot(1, 5, 2)
    plt.axis('off')
    f.add_subplot(2, 2, 2)
    plt.imshow(np.log10(i_image), origin="lower", cmap='hsv')
    # plt.imshow(cmap='Greys', origin='lower', alpha=0.5)
    plt.axis('off')
    plt.title("Initial Intensity Plot")
    f.add_subplot(2, 2, 3)
    plt.imshow(simulated_image, cmap="Greys", origin="lower")
    plt.title("Model")
    plt.axis('off')
    f.add_subplot(2, 2, 4)
    plt.imshow(np.log10(hiprmc.abs2(hiprmc.fourier_transform(simulated_image))), origin='lower', cmap='hsv')
    plt.axis('off')
    plt.title("Model Intensity Plot")
    plt.show()

    ######################################################
    ########### End Test Images ##########################
    ######################################################

def test_real():

    ######################################################
    ######### Uncomment for Uploading Real Data ##########
    ######################################################

    #Image upload
    image = np.load('cool_40p0C_ILC1_insitu_40pt0C_2m.edf.npy')
    print(image.shape)
    f_mask = np.load('shuai_data_mask.edf.npy')
    N = image.shape[0]
    image[image==np.nan]=0

    #set load of particles
    load = 0.10

    #random particles
    initial_particles = np.random.choice([0, 1], size=(N,N), p=[1.0-load, load])

    ######################################################
    ########### End Real Data Upload #####################
    ######################################################


    T = 40 # i_image.shape[0]  # temperature is on the same order as the image

    simulated_image = hiprmc.rmc(image, num_particles, N, T, iterations, movie=True)

    f = plt.figure(3)
    f.add_subplot(2, 2, 1)
    plt.imshow(image, cmap='Greys', origin="lower")
    plt.axis('off')
    plt.title("original image")
    f.add_subplot(1, 5, 2)
    plt.axis('off')
    #f.add_subplot(2, 2, 2)
    #original_intensity = np.log10(hiprmc.abs2(hiprmc.fourier_transform(i_image)))
    #plt.imshow(original_intensity, origin="lower", cmap='hsv')
    # plt.imshow(cmap='Greys', origin='lower', alpha=0.5)
    #plt.axis('off')
    #plt.title("Initial Intensity Plot")
    f.add_subplot(2, 2, 3)
    plt.imshow(simulated_image, cmap="Greys", origin="lower")
    plt.title("Model")
    plt.axis('off')
    f.add_subplot(2, 2, 4)
    plt.imshow(np.log10(hiprmc.abs2(hiprmc.fourier_transform(simulated_image))), origin='lower', cmap='hsv')
    plt.axis('off')
    plt.title("Model Intensity Plot")
    plt.show()


if __name__ == '__main__':
    test_simulated()