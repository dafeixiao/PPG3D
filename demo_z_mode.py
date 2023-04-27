
import numpy as np
import scipy.io as sio
import skimage.io as io
from sfs import noise_processing, psf_shifting
from PPG import PPG_z
import matplotlib.pyplot as plt


# load PSF zstack and process noise
psf_zstack = io.imread('./Data/psf_zstack.tif')
psfs = np.zeros(psf_zstack.shape)
for i in range(psf_zstack.shape[0]):
    psf = noise_processing(psf_zstack[i, :, :], thr=1.0)
    psf = (psf-psf.min())/(psf.max()-psf.min())
    psfs[i, :, :] = psf

# load axial positions
mat_dict = sio.loadmat('./Data/nfp_pos.mat')
dict_keys = list(mat_dict.keys())
zs = mat_dict[dict_keys[3]].flatten()  # um

# build the generator
psf_generator = PPG_z(psfs, zs)

x = 21.2  # um
y = -31  # um
z = 0.5  # um
psf_centered = psf_generator(z)  # no xy information here


# demonstration
plt.figure(12)
plt.imshow(psf_centered)
plt.title('centered PSF')
plt.show()

im_size = 600  # size in pixel
pixel_size = 0.13  # um, at the object plane
psf_xyz = psf_shifting(psf_centered, x, y, im_size, pixel_size)
plt.figure(13)
plt.imshow(psf_xyz)
plt.title('shifted PSF')
plt.show()
