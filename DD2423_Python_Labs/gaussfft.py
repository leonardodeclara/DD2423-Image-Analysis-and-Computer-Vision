import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def gaussfft(pic, t):
    x_dim,y_dim = pic.shape

    [x, y] = np.meshgrid(np.arange(-x_dim/2, x_dim/2), np.arange(-y_dim/2, y_dim/2))

    gaussian_kernel = 1 / (2 * np.pi * t) * np.exp(-(x**2 + y**2) / (2 * t))
    
    pic_hat = fft2(pic)
    gaussian_hat = fft2(gaussian_kernel)

    convolved_image = pic_hat * gaussian_hat


    inverted_pic = ifft2(convolved_image)
    out = fftshift(inverted_pic)

    return out.real
