import numpy as np
import numbers
from itertools import product
import torch

def check_grid(grid_or_shape, to_torch=True):
    if isinstance(grid_or_shape, tuple):
        shape = grid_or_shape
        slices = [slice(-(s // 2), -(s // 2) + s) for s in shape]
        grid = np.mgrid[slices]
    else:
        grid = grid_or_shape
        assert grid.shape[0] == len(grid.shape) - 1  # first dimension of grid must be ndim
    
    if isinstance(grid, np.ndarray) and to_torch:
        grid = torch.from_numpy(grid.astype('float32'))
    return grid


def simple_gabor(grid_or_shape, scale=0, xi0=np.pi * 3 / 4, sigma0=0.8, orientation=(1, 0)):
    """Creates a simple, isotropic-envelope Gabor filter. Mostly for testing purposes"""
    
    grid = check_grid(grid_or_shape)
    
    radii = np.linalg.norm(grid, axis=0)
    
    sigma0 = 0.8
    xi0 = np.pi * 3 / 4
    orientation = (1, 0)
    sigma = sigma0 * 2 ** scale
    xi = xi0 / 2 ** scale
    orientation = orientation / np.linalg.norm(orientation)
    
    gaussian = np.exp(-(radii / sigma) ** 2 * .5) / np.sqrt((2 * np.pi) ** 2 * sigma ** 4)
    
    orientation_grid = grid.T.dot(orientation).T
    wave = np.exp(1j * xi * orientation_grid)
    
    gabor = gaussian * wave
    return gabor


def bump(omega):
    
    mask = 1. * ((omega ** 2) < 1)
    g = np.exp(-omega ** 2 / (1 - omega ** 2 * mask))
    return mask * g


def radial_bump_angular_bump_fwavelet_3d(grid_or_shape, orientation, scale_center, angular_width, scale_width, device=None,
                                            ifftshift=True):

    if isinstance(orientation, np.ndarray):
        orientation = torch.from_numpy(orientation.astype('float32'))
    
    if device is None:
        device = orientation.device
    
    assert orientation.ndim == 1 and len(orientation) == 3

    orientation = orientation / torch.norm(orientation)
    z_axis = torch.from_numpy(np.array([0., 0., 1.]).astype('float32'))

    grid = check_grid(grid_or_shape, to_torch=True)

    if torch.dot(orientation, z_axis) < 1 - 1e-5:
        # if orientation is not z-axis, rotate grid
        aux_axis = torch.cross(orientation, z_axis)
        aux_axis = aux_axis / torch.norm(aux_axis)
        aux_z = torch.cross(orientation, aux_axis)
        basis = torch.stack((orientation, aux_z, aux_axis), dim=1)
        angle = torch.acos(torch.dot(orientation, z_axis))
        rot3db = torch.tensor([[torch.cos(-angle), -torch.sin(-angle), 0],
                                [torch.sin(-angle), torch.cos(-angle), 0],
                                [0., 0., 1.]])
        rot3d = torch.mm(basis, rot3db.mm(basis.T))
        assert torch.norm(rot3d.mv(orientation) - z_axis) < 1e-5

        grid = torch.matmul(grid.T, rot3d.T).T
    
    x, y, z = grid
    r = torch.norm(grid, dim=0)
    elevation = torch.acos(z / (r + 1e-6))

    angular_bump = bump(elevation / angular_width)


    radial_bump = bump((r - scale_center) / scale_width)

    fwavelet = radial_bump * angular_bump

    if ifftshift:
        return torch.fft.ifftshift(fwavelet)
    return fwavelet


def radial_bump_angular_bump_fwavelet_3d_bank(grid_or_shape, orientations, angular_width,
                                                J=None, Q=1, 
                                                high_freq_scale_width=None,
                                                high_freq_scale_center=None,
                                                ifftshift=True):
    
    grid = check_grid(grid_or_shape, to_torch=False)
    ndim = grid.shape[0]
    shape = grid.shape[1:]

    if orientations.shape[-1] != ndim:
        raise ValueError(f"grid of shape {grid.shape} implies dimensionality of {ndim}, but orientations of shape {orientations.shape} imply {orientations.shape[-1]}")
    if len(orientations.shape) != 2:
        raise ValueError(f"orientations must be 2-dimensional array, shape {orientations.shape} is not that")

    n_orientations = len(orientations)
    
    if high_freq_scale_width is None:
        high_freq_scale_width = min(shape) / 4.0 / Q
    if high_freq_scale_center is None:
        high_freq_scale_center = min(shape) / 2.0 - high_freq_scale_width

    if J is None:
        J = int(np.log2(min(shape)) - 2)
    scalings = 2 ** -(torch.arange(0, J * Q) / Q)
    scale_centers = high_freq_scale_center * scalings
    scale_widths = high_freq_scale_width * scalings

    #orientations, scale_centers = np.meshgrid(orientations, scale_centers)
    orientations = orientations[np.newaxis] * torch.ones((len(scalings), 1, 1))
    scale_centers = scale_centers[:, np.newaxis] * torch.ones((1, n_orientations))
    scale_widths = scale_widths[:, np.newaxis] * np.ones_like(scale_centers)

    fwavelets = [radial_bump_angular_bump_fwavelet_3d(grid, orientation, 
                                                        scale_center,
                                                        angular_width, 
                                                        scale_width,
                                                        ifftshift=ifftshift)
                    for orientation, scale_center, scale_width in
                    zip(orientations.reshape(-1, ndim), scale_centers.reshape(-1), scale_widths.reshape(-1))]
    fwavelets = torch.stack(fwavelets, dim=0)
    _, *s = fwavelets.shape
    fwavelets = fwavelets.reshape((scalings.shape[0], orientations.shape[1]) + tuple(s))
    return fwavelets


def compute_spherical_coordinate_orientations(main_orientation=(0, 0, 1), aux_orientation=None,
                        n_azimuths=8,
                        n_elevations=6,
                        half_sphere_azimuth=False,
                        half_sphere_elevation=False,
                        exclude_main_orientation=True):
    
    main_orientation = np.asarray(main_orientation, dtype='float').ravel()
    main_orientation /= np.linalg.norm(main_orientation)
    
    if aux_orientation is None:
        aux_orientation = np.linalg.svd(main_orientation.reshape(1, -1))[2][1]
    
    aux_orientation = np.asarray(aux_orientation, dtype='float').ravel()
    aux_orientation /= np.linalg.norm(aux_orientation)
    
    aux_orientation -= main_orientation * main_orientation.dot(aux_orientation)
    aux_orientation /= np.linalg.norm(aux_orientation)
    
    cross_orientation = np.cross(main_orientation, aux_orientation)
    
    max_azimuth = 2 * np.pi
    if half_sphere_azimuth:
        max_azimuth = np.pi
        
    azimuth_angles = max_azimuth * np.linspace(0., 1., n_azimuths, endpoint=False)
    
    equatorial_orientations = (np.cos(azimuth_angles[:, np.newaxis]) * aux_orientation + 
                               np.sin(azimuth_angles[:, np.newaxis]) * cross_orientation)
    
    
    if half_sphere_elevation:
        elevation_angles = np.pi / 2 * np.linspace(0., 1., n_elevations, endpoint=True)
    else:
        elevation_angles = np.pi * np.linspace(0., 1., n_elevations, endpoint=False)
    if exclude_main_orientation:
        elevation_angles = elevation_angles[1:]
    
    orientations = (np.cos(elevation_angles[:, np.newaxis, np.newaxis]) * main_orientation +
                    np.sin(elevation_angles[:, np.newaxis, np.newaxis]) * equatorial_orientations)
    return orientations



def radial_bump_angular_bump_fwavelet_3d_bank_order_2(grid_or_shape, orientations_order_1, angular_width,
                                                n_azimuths=4,
                                                n_elevations=5,
                                                half_sphere_azimuth=True,
                                                half_sphere_elevation=True,
                                                J=None, Q=1, 
                                                high_freq_scale_width=None,
                                                high_freq_scale_center=None,
                                                ifftshift=True):
    grid = check_grid(grid_or_shape, to_torch=True)
    ndim = len(grid.shape[1:])

    n_orientations_order_1 = len(orientations_order_1)
    all_fwavelets = []

    for l, orientation in enumerate(orientations_order_1):
        previous_orientation = orientations_order_1[l - 1] # wraps around when l == 0

        orientations_order_2 = compute_spherical_coordinate_orientations(main_orientation=orientation,
                                                                            aux_orientation=previous_orientation,
                                                                            n_azimuths=n_azimuths,
                                                                            n_elevations=n_elevations,
                                                                            half_sphere_azimuth=half_sphere_azimuth,
                                                                            half_sphere_elevation=half_sphere_elevation,
                                                                            exclude_main_orientation=True,
                                                                            )
        orientations_order_2 = torch.from_numpy(orientations_order_2.reshape(-1, ndim).astype('float32'))
        orientations_order_2 = torch.cat([orientation.reshape(1, ndim),
                                            orientations_order_2], dim=0)
        fwavelets = radial_bump_angular_bump_fwavelet_3d_bank(grid_or_shape, orientations_order_2, angular_width,
                                                J=J, Q=Q, 
                                                high_freq_scale_width=high_freq_scale_width,
                                                high_freq_scale_center=high_freq_scale_center,
                                                ifftshift=True)
        all_fwavelets.append(fwavelets)
    return all_fwavelets




def radial_bump_fwavelets(grid_or_shape, scale_center, scale_width, ifftshift=True):
    
    grid = check_grid(grid_or_shape, to_torch=False)
    
    radii = np.linalg.norm(grid, axis=0)
    
    radial_parts = bump(((radii[np.newaxis].T - scale_center) / scale_width).T)

    if ifftshift:
        return np.fft.ifftshift(radial_parts, axes=range(-grid[0].ndim, 0))
    return radial_parts
    




def radial_bump_fwavelet_bank(grid_or_shape, 
                                                J=None, Q=1, 
                                                high_freq_scale_width=None,
                                                high_freq_scale_center=None,
                                                ifftshift=True):
    
    grid = check_grid(grid_or_shape, to_torch=False)
    ndim = grid.shape[0]
    shape = grid.shape[1:]

    if high_freq_scale_width is None:
        high_freq_scale_width = min(shape) / 4.0 / Q
    if high_freq_scale_center is None:
        high_freq_scale_center = min(shape) / 2.0 - high_freq_scale_width

    if J is None:
        J = int(np.log2(min(shape)) - 2)
    scalings = 2 ** -(np.arange(0, J * Q) / Q)
    scale_centers = high_freq_scale_center * scalings
    scale_widths = high_freq_scale_width * scalings

    fwavelets = radial_bump_fwavelets(grid, 
                                            scale_centers,
                                            scale_widths,
                                            ifftshift=ifftshift)
    _, *s = fwavelets.shape
    fwavelets = fwavelets.reshape((scalings.shape[0], ) + tuple(s))
    return fwavelets



def spherical_harmonic_collection_scipy(grid_or_shape, Ls=(0, 1, 2, 3, 4), ifftshift=True):
    from scipy.special import sph_harm

    grid = check_grid(grid_or_shape, to_torch=False)
    shape = grid.shape[1:]
    ndim = grid.ndim - 1

    x, y, z = grid
    r = np.linalg.norm(grid, axis=0)

    theta = np.arccos(z / (r + 1e-19))
    phi = np.arctan2(y, x)
    all_harmonics = []

    for L in Ls:
        one_L_harmonics = np.zeros((2 * L + 1,) + shape, dtype='complex64')
        all_harmonics.append(one_L_harmonics)
        for i, m in enumerate(range(-L, L + 1)):
            harmonic = sph_harm(m, L, phi, theta)
            if ifftshift:
                harmonic = np.fft.ifftshift(harmonic)
            one_L_harmonics[i] = harmonic

    return all_harmonics


def spherical_harmonic_ring_bump_wavelets(grid_or_shape, 
                                                J=None, Q=1,
                                                Ls=(0, 1, 2, 3, 4),
                                                high_freq_scale_width=None,
                                                high_freq_scale_center=None,
                                                ifftshift=True,
                                                device=None,
                                                use_scipy=True):
    fring_bumps = radial_bump_fwavelet_bank(grid_or_shape,
                                            J=J, Q=Q,
                                            high_freq_scale_width=high_freq_scale_width,
                                            high_freq_scale_center=high_freq_scale_center,
                                            ifftshift=ifftshift).astype('float32')

    if use_scipy:
        harmonic_collection = spherical_harmonic_collection_scipy(grid_or_shape, Ls, ifftshift)
        output = [fring_bumps[:, np.newaxis] * harmonic[np.newaxis] for harmonic in harmonic_collection]
        return output
    
    else:
        raise NotImplementedError
    
    fring_bumps = torch.from_numpy(fring_bumps)
    if device is not None:
        fring_bumps = fring_bumps.to(device)
    
    harmonic_collection = spherical_harmonic_collection(grid_or_shape, Ls=Ls, ifftshift=ifftshift, device=device)

    output = [fring_bumps[:, np.newaxis] * harmonic.permute(3, 0, 1, 2)[np.newaxis] for harmonic in harmonic_collection]

    return output



def wavelets(fwavelets):


    # side_length = nc
    # fwavelets = wav.spherical_harmonic_ring_bump_wavelets((side_length,) * 3, J=3, Q=4, Ls=(0, 1, 2),
    #                                                  high_freq_scale_center=side_length / 4.,
    #                                                  high_freq_scale_width=side_length / 4., use_scipy=True)
    print(len(fwavelets), fwavelets[0].shape, fwavelets[1].shape, fwavelets[2].shape)
    exponents = [1., 0.5]

    for i_lhc in range(0, 1):
        print('LHC %i' % i_lhc)
        # read in halo catalog
        halos = Halos.Quijote_LHC_HR(i_lhc, z=zred)
        mesh = pm.paint(halos['Position'].compute())

        start = time.time()
        ps = FFTPower(mesh, mode='1d').power.data
        end = time.time()
        t0 = end-start 
        print("Time for PS : ", end-start)
        k, p = ps['k'], ps['power'].real
        fsignals = np.fft.fftn(mesh, axes=(-3, -2, -1))
        print(fsignals.shape)


        #tf version here
        all_filtered_signals = []
        summaries = []
        #fsignals = tf.signal.fft3d(mesh)
        fsignals = torch.fft.fftn(torch.tensor(mesh[...]), dim=(-3, -2, -1))
        start = time.time()
        for L, Lspace in enumerate(fwavelets):
            ffiltered_signals = fsignals[np.newaxis, np.newaxis, ...] * Lspace
            fshape = (Lspace.shape[0], Lspace.shape[1], nc, nc, nc) #ffiltered_signals.shape
            filtered_signals = np.zeros(fshape)
            #for i in range(fshape[0]):
            #    for j in range(fshape[1]):
            #        filtered_signals[i, j] = torch.fft.ifftn(ffiltered_signals[i, j], dim=(-3, -2, -1)).numpy()
            filtered_signals = torch.fft.ifftn(ffiltered_signals, dim=(-3, -2, -1)).numpy()
            # start = time.time()
            # for _ in range(3):
            #     filtered_signals = torch.fft.ifftn(ffiltered_signals, dim=(-3, -2, -1)).numpy()
            # print(time.time() - start)

            agg_filtered_signals = np.sqrt((np.abs(filtered_signals) ** 2).sum(-4))
            for q in exponents:
                summary = (agg_filtered_signals**q).mean(axis=(-3, -2, -1))
                summaries.append(summary)
            all_filtered_signals.append(agg_filtered_signals)

        all_filtered_signals = np.stack(all_filtered_signals, axis=2)
        #print(summaries)
