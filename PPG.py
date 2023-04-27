
# classes for PCA-based PSF Generator (PPG), written by Dafei Xiao, on 27.04.2023

import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from scipy import interpolate
from sfs import polynomial_surface_fitting
from math import pi
import itertools


class PPG_z(nn.Module):
    def __init__(self, psf_stack, z_list, vr=0.99):
        """
        PSF interpolation along z
        :param psf_stack: ndarray, rank3, n*h*w, with a certain normalization
        :param z_list: ndarray, rank1, n
        :param vr: variance ratio for PCA
        """
        super().__init__()
        self.psf_stack = psf_stack
        self.z_list = z_list
        self.zmin, self.zmax = z_list.min(), z_list.max()
        self.psf_h = psf_stack.shape[1]
        self.psf_w = psf_stack.shape[2]
        self.num_z = psf_stack.shape[0]
        X = psf_stack.reshape(self.num_z, -1)
        self.pca = PCA(vr)  # PCA class with specified variance ratio
        self.X_new = self.pca.fit_transform(X)
        self.new_dimensionality = self.X_new.shape[1]
        self.pcs = self.pca.components_.reshape(-1, self.psf_h, self.psf_w)  # principal components

    def forward(self, z):
        """
        PSF generator
        :param z: axial position to be interpolated
        :return: PSF at z
        """
        if z < self.zmin or z > self.zmax:
            raise Exception('Out of z range')
        c_interp = np.zeros((1, self.new_dimensionality))
        for col_idx in range(self.new_dimensionality):  # interpolation
            c_col = self.X_new[:, col_idx]
            fun = interpolate.interp1d(self.z_list, c_col, kind='cubic')
            c_interp[0, col_idx] = fun(z)
        psf_z = self.pca.inverse_transform(c_interp).reshape(self.psf_h, self.psf_w)
        psf_z = np.abs(psf_z)  # ensure the non-negativity of PSF
        return psf_z


class PPG_xyz(nn.Module):
    def __init__(self, all_psfs, xy_pos, z_pos):
        """
        PSF interpolation/fitting class
        :param all_psfs: ndarray, rank 4, [xy_num, z_num, psf_size h, psf_size w]
        :param xy_pos: ndarray, rank 2, [xy_num, 2], physical coordinates at the object plane um
        :param z_pos: ndarray, rank 1, [z_num, ], z means nfp or z of emitters um
        """
        super().__init__()
        self.all_psfs = all_psfs  # rank4
        self.xy_num, self.z_num, self.psf_size = all_psfs.shape[0], all_psfs.shape[1], all_psfs.shape[2:]
        self.pca_variance = 0.95  # the amount of explained variance in PCA
        self.pca_list = []  # save PCA objects at all axial planes
        self.pca_X_new_list = []  # new data after dimensionality reduction in a new space
        for i in range(self.z_num):
            X = all_psfs[:, i, :, :].reshape(self.xy_num, -1)
            pca = PCA(n_components=self.pca_variance)
            X_new = pca.fit_transform(X)
            self.pca_list.append(pca)
            self.pca_X_new_list.append(X_new)
        self.xy_pos = xy_pos
        if xy_pos.shape[0] != self.xy_num:
            raise Exception("xy number mismatch in psf data and position data!")
        self.x_range = [np.min(xy_pos[:, 0]), np.max(xy_pos[:, 0])]  # um
        self.y_range = [np.min(xy_pos[:, 1]), np.max(xy_pos[:, 1])]  # um
        self.z_pos = z_pos  # um
        self.z_range = [np.min(z_pos), np.max(z_pos)]
        self.num_pick1 = 7  # first range in lateral point selection
        self.num_pick2 = 13  # second range in lateral point selection, when pick1 fails
        self.fitting_num_xy = 3  # point number in fitting in a lateral plane, 2D
        self.fitting_order = 1  # order of the 2D fitting
        self.fitting_num_z = 2  # point number in fitting along z, 1D
        self.nn_flag = True  # PSF's non-negative flag, necessary for a small number of calibration positions
        self.max_r = np.max(np.sqrt(np.sum(xy_pos**2, axis=1)))

        print(f'x range in calibration: {self.x_range} [um]')
        print(f'y range in calibration: {self.y_range} [um]')
        print(f'axial range in calibration: {self.z_range} [um]')
        print(f'maximum radius: {self.max_r} [um]')

    def fitting_xy(self, c, xys, xy, order):  # 2D fitting
        """
        2D fitting regarding c (coefficients after PCA, data X in new space)
        :param c: ndarray, rank 2, [n_samples, n_features], coefficients
        :param xys: ndarray, rank 2, [n_samples, 2], known/selected xy coordinates
        :param xy: ndarray, rank 2, [1, 2], prediction position, one at each time
        :param order: int, fitting order, from 1 to 4
        :return: c_fitted, ndarray, rank2, [1, n_features]
        """
        c_fitted = np.zeros((1, c.shape[1]))  # coefficients to be interpolated
        for col_idx in range(c.shape[1]):
            data = np.c_[xys, c[:, col_idx]]
            z = polynomial_surface_fitting(data, order, xy)
            c_fitted[0, col_idx] = z
        return c_fitted

    def interpolation_axial(self, c, zs, z):
        """
        1D interpolation along axis
        :param c: ndarray, rank 2, [n_samples, n_features], coefficients
        :param zs: ndarray, rank1, [num_z, ], known z positions
        :param z: scalar, z for prediction,
        :return: interpolated c, ndarray, rank 2, [1, n_features], one at each time
        """
        c_interp = np.zeros((1, c.shape[1]))  # coefficients to be interpolated
        for col_idx in range(c.shape[1]):
            c_col = c[:, col_idx]
            f = interpolate.interp1d(zs, c_col)
            c_ = f(z)
            c_interp[0, col_idx] = c_
        return c_interp

    def select_points(self, xy):
        """
        find calibration points around target position xy
        :param xy: ndarray, rank 2, [1,2], target xy position
        :return: tuple, [num_points, ], indices of chosen points
        """
        distances_xy = np.sum((self.xy_pos - xy) ** 2, axis=1)  # distances from target xy to all calibration points
        self.min_idx = np.argmin(distances_xy)  # find the index of the closest point, for possible negative cases later
        indices_xy = np.argpartition(distances_xy,
                                     self.num_pick1)[:self.num_pick1]  # first selection regarding distance
        point_num = self.fitting_num_xy
        all_combinations = list(itertools.combinations(indices_xy, point_num))  # choose
        angle_variance_list = []  # choose according to angle variance, can also be distance of the triangle center to the desired point
        for combination in all_combinations:
            vectors = self.xy_pos[combination, :] - xy
            angles = np.sort(np.angle(vectors[:, 0] + 1j*vectors[:, 1]))  # [-pi, pi]
            target_mean_angle = 2*pi/point_num
            angle_variance = np.sum(((angles[1:]-angles[:-1])-target_mean_angle)**2)/(point_num-1)
            angle_variance_list.append(angle_variance)
        final_idx = all_combinations[np.argmin(angle_variance_list)]  # choose regarding angle variance/variation
        xys1 = self.xy_pos[final_idx, :]
        xmin, xmax = xys1[:, 0].min(), xys1[:, 0].max()
        ymin, ymax = xys1[:, 1].min(), xys1[:, 1].max()
        if xy[0, 0] > xmin and xy[0, 0] < xmax and xy[0, 1] > ymin and xy[0, 1] < ymax:  # is it a good selection?
            pass  # yes
        else:  # no, increase searching range
            indices_xy = np.argpartition(distances_xy, self.num_pick2)[:self.num_pick2]
            all_combinations = list(itertools.combinations(indices_xy, point_num))  # choose
            angle_variance_list = []
            for combination in all_combinations:
                vectors = self.xy_pos[combination, :] - xy
                angles = np.sort(np.angle(vectors[:, 0] + 1j * vectors[:, 1]))  # [-pi, pi]
                target_mean_angle = 2 * pi / point_num
                angle_variance = np.sum(((angles[1:] - angles[:-1]) - target_mean_angle) ** 2) / (point_num - 1)
                angle_variance_list.append(angle_variance)
            final_idx = all_combinations[np.argmin(angle_variance_list)]
        return final_idx

    def forward(self, x, y, z):
        """
        PSF interpolation at xyz position
        :param x: scalar, pixel coordinate at image plane
        :param y: scalar, pixel coordinate at image plane
        :param z: scalar, axial coordinate for object, z of an emitter or nfp, unit: um
        :return: ndarray, rank2, [psf_size, psf_size], interpolated PSF
        """
        # find xy positions used for fitting, how many laterally and axially
        target_xy = np.array([[x, y]])
        indices_xy = self.select_points(target_xy)  # It matters
        xys = self.xy_pos[indices_xy, :]
        distances_z = (self.z_pos-z)**2
        indices_z = np.argpartition(distances_z, self.fitting_num_z)[:self.fitting_num_z]
        zs = self.z_pos[indices_z]
        psfs_axial = np.zeros((self.fitting_num_z, self.psf_size[0]**2))
        for idx, idx_z in enumerate(indices_z):
            c = self.pca_X_new_list[idx_z][indices_xy, :]
            c_fitted = self.fitting_xy(c, xys, target_xy, self.fitting_order)
            fitted_psf = self.pca_list[idx_z].inverse_transform(c_fitted)
            psfs_axial[idx, :] = fitted_psf[0, :]
        # non-negativity test
        if np.abs(psfs_axial.min()) > 0.1 * psfs_axial.max():  # fail
            self.nn_flag = False
            psfs_axial = self.all_psfs[self.min_idx, indices_z, :, :].reshape(len(indices_z), -1)
        else:
            self.nn_flag = True

        pca_z = PCA()
        c = pca_z.fit_transform(psfs_axial)
        c_interpolated = self.interpolation_axial(c, zs, z)  # axial interpolation
        psf = pca_z.inverse_transform(c_interpolated).reshape(self.psf_size)
        psf = np.abs(psf)  # make it non-negative
        return psf