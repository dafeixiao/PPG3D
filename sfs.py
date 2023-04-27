# supplementary functions
import numpy as np
from skimage.morphology import erosion, dilation
import scipy


def noise_processing(im, corner_size=10, thr=2.0):
    """
    remove part of noise in a measured image
    :param im: image
    :param corner_size: pixels at each corner as noise references
    :param thr: values below thr*standard deviation are set to 0
    :return: the processed image
    """
    tmp = np.concatenate(
        (np.concatenate((im[:corner_size, :corner_size], im[-corner_size:, :corner_size]), axis=0),
         np.concatenate((im[:corner_size, -corner_size:], im[-corner_size:, -corner_size:]), axis=0)),
        axis=1)
    mean_value = np.mean(tmp)
    std_value = np.std(tmp)
    im0 = im-mean_value
    mask = (im0 > thr*std_value)
    cross = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])
    mask = erosion(mask, cross)
    mask = dilation(mask, cross)
    im1 = im0*mask
    im1 = im1.astype(tmp.dtype)
    return im1


def polynomial_surface_fitting(data, order, xy):
    """
    Polynomial surface fitting
    :param data: ndarray, rank 2, [~,3], the first, second and third column corresponds to x, y and z respectively
    :param order: int, 1, 2, 3, or 4
    :param xy: ndarray, rank 2, [~, 2], the first, second column are x and y coordinates to predict
    :return: ndarray, rank1, [~,], ? predicted z at (x_col, y_col)
    """
    x, y = xy[:, 0], xy[:, 1]
    xdata, ydata, zdata = data[:, 0], data[:, 1], data[:, 2]
    constant = np.ones(data.shape[0])
    if order == 1:
        A = np.c_[constant, xdata, ydata]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y], C)
        return z
    elif order==2:
        A = np.c_[constant, xdata, ydata, xdata*ydata, xdata**2, ydata**2]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y, x*y, x**2, y**2], C)
        return z
    elif order==3:
        A = np.c_[constant, xdata, ydata, xdata * ydata, xdata ** 2, ydata ** 2, xdata*ydata**2, xdata**2*ydata,
                  xdata**3, ydata**3]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y, x * y, x ** 2, y ** 2, x*y**2, x**2*y, x**3, y**3], C)
        return z
    elif order==4:
        A = np.c_[constant, xdata, ydata, xdata * ydata, xdata ** 2, ydata ** 2, xdata * ydata ** 2, xdata ** 2 * ydata,
                  xdata ** 3, ydata ** 3, xdata*ydata**3, xdata**2*ydata**2, xdata**3*ydata, xdata**4, ydata**4]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y, x * y, x ** 2, y ** 2, x * y ** 2, x ** 2 * y, x ** 3, y ** 3,
                   x*y**3, x**2*y**2, x**3*y, x**4, y**4], C)
        return z
    elif order==5:
        A = np.c_[constant,
                  xdata, ydata,
                  xdata * ydata, xdata ** 2, ydata ** 2,
                  xdata * ydata ** 2, xdata ** 2 * ydata, xdata ** 3, ydata ** 3,
                  xdata * ydata ** 3, xdata ** 2 * ydata ** 2, xdata ** 3 * ydata, xdata ** 4, ydata ** 4,
                  ydata**5, xdata*ydata**4, xdata**2*ydata**3, xdata**3*ydata**2, xdata**4*ydata**1, xdata ** 5]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y, x * y, x ** 2, y ** 2, x * y ** 2, x ** 2 * y, x ** 3, y ** 3,
                         x * y ** 3, x ** 2 * y ** 2, x ** 3 * y, x ** 4, y ** 4,
                         y ** 5, x*y ** 4, x**2*y ** 3, x**3*y ** 2, x**4*y ** 1, x**5], C)

        return z
    else:
        print('Order should be smaller than 4')


def psf_shifting(psf_centered, x, y, im_size, pixel_size):
    """
    shift the centered PSF to the specified xy position in a memory-efficient way
    :param psf_centered: ccentered psf, from interpolation here
    :param x: physical x coordinate at object plane, um
    :param y: physical y coordinate at object plane, um
    :param im_size: final size of the shifted psf image
    :param pixel_size: pixel size
    :return: a PSF image
    """

    psf_final = np.zeros((im_size, im_size))  # image size of the final PSF

    interpolation_size = psf_centered.shape[0]
    x_, y_ = x / pixel_size, y / pixel_size  # pixel location
    r_start_ = y_ + im_size / 2 - interpolation_size / 2
    r_start = int(np.floor(y_ + im_size / 2 - interpolation_size / 2))
    r_shift = r_start_ - r_start
    r_end = r_start + interpolation_size
    c_start_ = x_ + im_size / 2 - interpolation_size / 2
    c_start = int(np.floor(x_ + im_size / 2 - interpolation_size / 2))
    c_shift = c_start_ - c_start
    c_end = c_start + interpolation_size
    psf = np.abs(scipy.ndimage.shift(psf_centered, (r_shift, c_shift)))  # sub-pixel shifting
    if r_start < 0 or c_start < 0 or r_end > im_size or c_end > im_size:
        raise Exception(f'OUT OF RANGE!')

    # this is better considering the superposition of many PSFs in a large FOV
    psf_final[r_start: r_end, c_start: c_end] = psf

    return psf_final


