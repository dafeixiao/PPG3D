
import scipy.io as sio
import numpy as np
from PPG import PPG_xyz
import matplotlib.pyplot as plt
from sfs import psf_shifting


# load psf stacks, 4 pieces, The data division is for storage and uploading.
psfs_1 = sio.loadmat('./Data/psf_stacks_1.mat')['psf_stacks_1']
psfs_1 = psfs_1.astype(dtype=np.double)
psfs_2 = sio.loadmat('./Data/psf_stacks_2.mat')['psf_stacks_2']
psfs_2 = psfs_2.astype(dtype=np.double)
psfs_3 = sio.loadmat('./Data/psf_stacks_3.mat')['psf_stacks_3']
psfs_3 = psfs_3.astype(dtype=np.double)
psfs_4 = sio.loadmat('./Data/psf_stacks_4.mat')['psf_stacks_4']
psfs_4 = psfs_4.astype(dtype=np.double)
all_psfs = np.concatenate((psfs_1, psfs_2, psfs_3, psfs_4), axis=1)


# load xy_pos and z_pos
xy_pos = sio.loadmat('./Data/xy_pos.mat')['xy_pos'].astype(np.double)  # um
z_pos = sio.loadmat('./Data/z_pos.mat')['z_pos'].flatten().astype(np.double)  # um

# build the generator
psf_generator = PPG_xyz(all_psfs, xy_pos, z_pos)

# apply the generator
x = 21.2  # um
y = -31  # um
z = 0.5  # um
psf_centered = psf_generator(x, y, z)

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
