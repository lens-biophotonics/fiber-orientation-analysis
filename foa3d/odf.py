import numpy as np
import scipy as sp
from numba import njit
from skimage.transform import resize

from foa3d.utils import normalize_image, transform_axes


def compute_disarray(fbr_vec):
    """
    Compute local angular disarray, i.e. the misalignment degree of nearby orientation vectors
    with respect to the mean direction (Giardini et al., Front. Physiol. 2021).

    Parameters
    ----------
    fbr_vec: numpy.ndarray (shape=(N,3), dtype=float)
        array of fiber orientation vectors
        (reshaped super-voxel of shape=(Nz,Ny,Nx), i.e. N=Nz*Ny*Nx)

    Returns
    -------
    disarray: float32
        local angular disarray
    """
    fbr_vec = np.delete(fbr_vec, np.all(fbr_vec == 0, axis=-1), axis=0)
    disarray = (1 - np.linalg.norm(np.mean(fbr_vec, axis=0))).astype(np.float32)

    return disarray


@njit(cache=True)
def compute_fiber_angles(fbr_vec, norm):
    """
    Estimate the spherical coordinates (φ azimuth and θ polar angles)
    of the fiber orientation vectors returned by the Frangi filtering stage
    (all-zero background vectors are excluded).

    Parameters
    ----------
    fbr_vec: numpy.ndarray (shape=(N,3), dtype=float)
        array of fiber orientation vectors
        (reshaped super-voxel of shape=(Nz,Ny,Nx), i.e. N=Nz*Ny*Nx)

    norm: numpy.ndarray (shape=(N,), dtype=float)
        2-norm of fiber orientation vectors

    Returns
    -------
    phi: numpy.ndarray (shape=(N,), dtype=float)
        fiber azimuth angle [rad]

    theta: numpy.ndarray (shape=(N,), dtype=float)
        fiber polar angle [rad]
    """

    fbr_vec = fbr_vec[norm > 0, :]
    phi = np.arctan2(fbr_vec[:, 1], fbr_vec[:, 2])
    theta = np.arccos(fbr_vec[:, 0] / norm[norm > 0])

    return phi, theta


def compute_odf_map(fbr_vec, odf, odi_pri, odi_sec, odi_tot, odi_anis, disarray, vec_tensor_eigen, vxl_side, odf_norm,
                    odf_deg=6, vxl_thr=0.5, vec_thr=0.000001):
    """
    Compute the spherical harmonics coefficients iterating over super-voxels
    of fiber orientation vectors.

    Parameters
    ----------
    fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float)
        fiber orientation vectors

    odf: NumPy memory-map object (axis order=(X,Y,Z,C), dtype=float32)
        initialized array of ODF spherical harmonics coefficients

    odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        primary orientation dispersion index

    odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        secondary orientation dispersion index

    odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        total orientation dispersion index

    odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        orientation dispersion index anisotropy

    disarray: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        local angular disarray

    vec_tensor_eigen: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        initialized array of orientation tensor eigenvalues

    vxl_side: int
        side of the ODF super-voxel [px]

    odf_norm: numpy.ndarray (dtype: float)
        2D array of spherical harmonics normalization factors

    odf_deg: int
        degrees of the spherical harmonics series expansion

    vxl_thr: float
        minimum relative threshold on the sliced voxel volume

    vec_thr: float
        minimum relative threshold on non-zero orientation vectors

    Returns
    -------
    odf: numpy.ndarray (axis order=(X,Y,Z,C), dtype=float32)
        volumetric map of real-valued spherical harmonics coefficients
    """

    # iterate over ODF super-voxels
    ref_vxl_size = min(vxl_side, fbr_vec.shape[0]) * vxl_side**2
    for z in range(0, fbr_vec.shape[0], vxl_side):
        zmax = z + vxl_side

        for y in range(0, fbr_vec.shape[1], vxl_side):
            ymax = y + vxl_side

            for x in range(0, fbr_vec.shape[2], vxl_side):
                xmax = x + vxl_side

                # slice orientation voxel
                vec_vxl = fbr_vec[z:zmax, y:ymax, x:xmax, :]
                zerovec = np.count_nonzero(np.all(vec_vxl == 0, axis=-1))
                sli_vxl_size = np.prod(vec_vxl.shape[:-1])

                # skip boundary voxels and voxels without enough non-zero orientation vectors
                if sli_vxl_size / ref_vxl_size > vxl_thr and 1 - zerovec / sli_vxl_size > vec_thr:

                    # flatten orientation supervoxel
                    vec_arr = vec_vxl.ravel()

                    # compute ODF and orientation tensor eigenvalues
                    zv, yv, xv = z // vxl_side, y // vxl_side, x // vxl_side
                    odf[zv, yv, xv, :] = fiber_vectors_to_sph_harm(vec_arr, odf_deg, odf_norm)
                    vec_tensor_eigen[zv, yv, xv, :] = compute_vec_tensor_eigen(vec_arr)
                    disarray[zv, yv, xv] = compute_disarray(vec_arr)

    # compute slice-wise dispersion and anisotropy parameters
    compute_orientation_dispersion(vec_tensor_eigen, odi_pri, odi_sec, odi_tot, odi_anis)

    # keep a black background
    mask_orientation_dispersion([odi_pri, odi_sec, odi_tot, odi_anis], vec_tensor_eigen)

    # transform axes
    odf = transform_axes(odf, swapped=(0, 2), flipped=(1, 2))

    return odf


@njit(cache=True)
def compute_orientation_dispersion(vec_tensor_eigen, odi_pri, odi_sec, odi_tot, odi_anis):
    """
    Compute orientation dispersion parameters.

    Parameters
    ----------
    vec_tensor_eigen: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        orientation tensor eigenvalues
        computed from an ODF super-voxel

    odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        primary orientation dispersion index

    odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        secondary orientation dispersion index

    odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        total orientation dispersion index

    odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        orientation dispersion index anisotropy

    Returns
    -------
    odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        primary orientation dispersion index

    odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        secondary orientation dispersion index

    odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        total orientation dispersion index

    odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        orientation dispersion index anisotropy
    """

    k1 = np.abs(vec_tensor_eigen[..., 2])
    k2 = np.abs(vec_tensor_eigen[..., 1])
    k3 = np.abs(vec_tensor_eigen[..., 0])
    diff = np.abs(vec_tensor_eigen[..., 1] - vec_tensor_eigen[..., 0])

    # primary dispersion (0.3183098861837907 = 1/π)
    odi_pri[:] = (0.5 - 0.3183098861837907 * np.arctan2(k1, k2)).astype(np.float32)

    # secondary dispersion
    odi_sec[:] = (0.5 - 0.3183098861837907 * np.arctan2(k1, k3)).astype(np.float32)

    # total dispersion
    odi_tot[:] = (0.5 - 0.3183098861837907 * np.arctan2(k1, np.sqrt(np.abs(np.multiply(k2, k3))))).astype(np.float32)

    # dispersion anisotropy
    odi_anis[:] = (0.5 - 0.3183098861837907 * np.arctan2(k1, diff)).astype(np.float32)


@njit(cache=True)
def compute_real_sph_harm(degree, order, phi, sin_theta, cos_theta, norm_factors):
    """
    Estimate the coefficients of the real spherical harmonics series expansion
    as described by Alimi et al. (Medical Image Analysis, 2020).

    Parameters
    ----------
    degree: int
        degree index of the spherical harmonics expansion

    order: int
        order index of the spherical harmonics expansion

    phi: float
        azimuth angle [rad]

    sin_theta: float
        polar angle sine

    cos_theta: float
        polar angle cosine

    norm_factors: numpy.ndarray (dtype: float)
        normalization factors

    Returns
    -------
    real_sph_harm: float
        real-valued spherical harmonic coefficient
    """

    if degree == 0:
        real_sph_harm = norm_factors[0, 0]
    elif degree == 2:
        real_sph_harm = sph_harm_degree_2(order, phi, sin_theta, cos_theta, norm_factors[1, :])
    elif degree == 4:
        real_sph_harm = sph_harm_degree_4(order, phi, sin_theta, cos_theta, norm_factors[2, :])
    elif degree == 6:
        real_sph_harm = sph_harm_degree_6(order, phi, sin_theta, cos_theta, norm_factors[3, :])
    elif degree == 8:
        real_sph_harm = sph_harm_degree_8(order, phi, sin_theta, cos_theta, norm_factors[4, :])
    elif degree == 10:
        real_sph_harm = sph_harm_degree_10(order, phi, sin_theta, cos_theta, norm_factors[5, :])
    else:
        raise ValueError("\n  Invalid degree of the spherical harmonics series expansion!!!")

    return real_sph_harm


def compute_vec_tensor_eigen(fbr_vec):
    """
    Compute the eigenvalues of the 3x3 orientation tensor
    obtained from a reshaped super-voxel of fiber orientation vectors.

    Parameters
    ----------
    fbr_vec: numpy.ndarray (shape=(N,3), dtype=float)
        fiber orientation vectors
        (reshaped super-voxel of shape=(Nz,Ny,Nx), i.e. N=Nz*Ny*Nx)

    Returns
    -------
    vec_tensor_eigen: numpy.ndarray (shape=(3,), dtype=float32)
        orientation tensor eigenvalues in ascending order
    """

    fbr_vec = np.delete(fbr_vec, np.all(fbr_vec == 0, axis=-1), axis=0)
    fbr_vec.shape = (-1, 3)
    t = np.zeros((3, 3))
    for v in fbr_vec:
        t += np.outer(v, v)

    vec_tensor_eigen = sp.linalg.eigh(t, eigvals_only=True).astype(np.float32)

    return vec_tensor_eigen


factorial_lut = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype=np.double)


@njit(cache=True)
def factorial(n):
    """
    Retrieve factorial using pre-computed LUT.

    Parameters
    ----------
    n: int
        integer number (max: 20)

    Returns
    -------
    f: int
        factorial
    """
    if n > 20:
        raise ValueError

    return factorial_lut[n]


@njit(cache=True)
def fiber_angles_to_sph_harm(phi, theta, degrees, norm_factors, ncoeff):
    """
    Generate the real-valued symmetric spherical harmonics series expansion
    from fiber φ azimuth and θ polar angles,
    i.e. the spherical coordinates of the fiber orientation vectors.

    Parameters
    ----------
    phi: numpy.ndarray (shape=(N,), dtype=float)
        fiber azimuth angles [rad]
        (reshaped super-voxel of shape=(Nz,Ny,Nx), i.e. N=Nz*Ny*Nx)

    theta: numpy.ndarray (shape=(N,), dtype=float)
        fiber polar angle [rad]
        (reshaped super-voxel of shape=(Nz,Ny,Nx), i.e. N=Nz*Ny*Nx)

    degrees: int
        degrees of the spherical harmonics expansion

    norm_factors: numpy.ndarray (dtype: float)
        normalization factors

    ncoeff: int
        number of spherical harmonics coefficients

    Returns
    -------
    real_sph_harm: numpy.ndarray (shape=(ncoeff,), dtype=float)
        array of real-valued spherical harmonics coefficients
        building the spherical harmonics series expansion
    """

    real_sph_harm = np.zeros(ncoeff)
    i = 0
    for n in range(0, degrees + 1, 2):
        for m in range(-n, n + 1, 1):
            for j, (p, t) in enumerate(zip(phi, theta)):
                real_sph_harm[i] += compute_real_sph_harm(n, m, p, np.sin(t), np.cos(t), norm_factors)
            i += 1

    real_sph_harm /= phi.size

    return real_sph_harm


def fiber_vectors_to_sph_harm(fbr_vec, degrees, norm_factors):
    """
    Generate the real-valued symmetric spherical harmonics series expansion
    from the fiber orientation vectors returned by the Frangi filter stage.

    Parameters
    ----------
    fbr_vec: numpy.ndarray (shape=(N,3), dtype=float)
        array of fiber orientation vectors
        (reshaped super-voxel of shape=(Nz,Ny,Nx), i.e. N=Nz*Ny*Nx)

    degrees: int
        degrees of the spherical harmonics expansion

    norm_factors: numpy.ndarray (dtype: float)
        normalization factors

    Returns
    -------
    real_sph_harm: numpy.ndarray (shape=(ncoeff,), dtype=float)
        real-valued spherical harmonics coefficients
    """

    fbr_vec.shape = (-1, 3)
    ncoeff = get_sph_harm_ncoeff(degrees)

    norm = np.linalg.norm(fbr_vec, axis=-1)
    if np.sum(norm) < np.sqrt(fbr_vec.shape[0]):
        return np.zeros(ncoeff)

    phi, theta = compute_fiber_angles(fbr_vec, norm)

    real_sph_harm = fiber_angles_to_sph_harm(phi, theta, degrees, norm_factors, ncoeff)

    return real_sph_harm


def generate_odf_background(bg_img, bg_mrtrix_mmap, vxl_side):
    """
    Generate the downsampled background image required
    to visualize the 3D ODF map in MRtrix3.

    Parameters
    ----------
    bg_img: NumPy memory-map object or HDF5 dataset
                  (axis order=(Z,Y,X), dtype=uint8; or axis order=(Z,Y,X,C), dtype=float32)
        fiber volume image or orientation vector volume
        to be used as the MRtrix3 background image

    bg_mrtrix_mmap: NumPy memory-map object or HDF5 dataset (axis order=(X,Y,Z), dtype=uint8)
        initialized background array for ODF visualization in MRtrix3

    vxl_side: int
        side of the ODF super-voxel [px]

    Returns
    -------
    bg_mrtrix_mmap: NumPy memory-map object or HDF5 dataset (axis order=(X,Y,Z), dtype=uint8)
        downsampled background for ODF visualization in MRtrix3
    """

    # get shape of new downsampled array
    new_shape = bg_mrtrix_mmap.shape[:-1]

    # normalize
    if bg_img.ndim == 3:
        bg_img = normalize_image(bg_img)

    # loop over z-slices, and resize them
    for z in range(0, bg_img.shape[0], vxl_side):
        if bg_img.ndim == 3:
            tmp_slice = np.mean(bg_img[z:z + vxl_side, ...], axis=0)
        elif bg_img.ndim == 4:
            tmp_slice = 255.0 * np.sum(np.abs(bg_img[z, ...]), axis=-1)
            tmp_slice = np.where(tmp_slice <= 255.0, tmp_slice, 255.0)

        tmp_slice = transform_axes(tmp_slice, swapped=(0, 1), flipped=(0, 1))
        bg_mrtrix_mmap[..., z // vxl_side] = \
            resize(tmp_slice, output_shape=new_shape, anti_aliasing=True, preserve_range=True)


@njit(cache=True)
def get_sph_harm_ncoeff(degrees):
    """
    Get the number of coefficients of the real spherical harmonics series
    expansion.

    Parameters
    ----------
    degrees: int
        degrees of the spherical harmonics series expansion

    Returns
    -------
    ncoeff: int
        number of spherical harmonics coefficients
    """
    ncoeff = (2 * (degrees // 2) + 1) * ((degrees // 2) + 1)

    return ncoeff


@njit(cache=True)
def get_sph_harm_norm_factors(degrees):
    """
    Estimate the normalization factors of the real spherical harmonics series
    expansion.

    Parameters
    ----------
    degrees: int
        degrees of the spherical harmonics series expansion

    Returns
    -------
    norm_factors: numpy.ndarray (dtype: float)
        2D array of spherical harmonics normalization factors
    """

    norm_factors = np.zeros(shape=(degrees + 1, 2 * degrees + 1))
    for n in range(0, degrees + 1, 2):
        for m in range(0, n + 1, 1):
            norm_factors[n, m] = norm_factor(n, m)

    norm_factors = norm_factors[::2]

    return norm_factors


def mask_orientation_dispersion(odi_lst, vec_tensor_eigen):
    """
    Keep a black background where no fiber orientation vector
    is available.

    Parameters
    ----------
    odi_lst: list
        list including the ODI maps estimated
        in compute_orientation_dispersion()

    vec_tensor_eigen: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        orientation tensor eigenvalues
        computed from an ODF super-voxel

    Returns
    -------
    None
    """

    bg_msk = np.all(vec_tensor_eigen == 0, axis=-1)

    for odi in odi_lst:
        odi[bg_msk] = 0


@njit(cache=True)
def norm_factor(n, m):
    """
    Compute the normalization factor of the term of degree n and order m
    of the real-valued spherical harmonics series expansion.

    Parameters
    ----------
    n: int
        degree index

    m: int
        order index

    Returns
    -------
    nf: float
        normalization factor
    """

    if m == 0:
        nf = np.sqrt((2 * n + 1) / (4 * np.pi))
    else:
        nf = (-1)**m * np.sqrt(2) * np.sqrt(((2 * n + 1) / (4 * np.pi) *
                                             (factorial(n - np.abs(m)) / factorial(n + np.abs(m)))))

    return nf


@njit(cache=True)
def sph_harm_degree_2(order, phi, sin_theta, cos_theta, norm):
    if order == -2:
        return norm[2] * 3 * sin_theta**2 * np.sin(2 * phi)
    elif order == -1:
        return norm[1] * 3 * sin_theta * cos_theta * np.sin(phi)
    elif order == 0:
        return norm[0] * 0.5 * (3 * cos_theta ** 2 - 1)
    elif order == 1:
        return norm[1] * 3 * sin_theta * cos_theta * np.cos(phi)
    elif order == 2:
        return norm[2] * 3 * sin_theta**2 * np.cos(2 * phi)


@njit(cache=True)
def sph_harm_degree_4(order, phi, sin_theta, cos_theta, norm):
    if order == -4:
        return norm[4] * 105 * sin_theta**4 * np.sin(4 * phi)
    elif order == -3:
        return norm[3] * 105 * sin_theta**3 * cos_theta * np.sin(3 * phi)
    elif order == -2:
        return norm[2] * 7.5 * sin_theta**2 * (7 * cos_theta ** 2 - 1) * np.sin(2 * phi)
    elif order == -1:
        return norm[1] * 2.5 * sin_theta * (7 * cos_theta ** 3 - 3 * cos_theta) * np.sin(phi)
    elif order == 0:
        return norm[0] * 0.125 * (35 * cos_theta ** 4 - 30 * cos_theta ** 2 + 3)
    elif order == 1:
        return norm[1] * 2.5 * sin_theta * (7 * cos_theta ** 3 - 3 * cos_theta) * np.cos(phi)
    elif order == 2:
        return norm[2] * 7.5 * sin_theta**2 * (7 * cos_theta ** 2 - 1) * np.cos(2 * phi)
    elif order == 3:
        return norm[3] * 105 * sin_theta**3 * cos_theta * np.cos(3 * phi)
    elif order == 4:
        return norm[4] * 105 * sin_theta**4 * np.cos(4 * phi)


@njit(cache=True)
def sph_harm_degree_6(order, phi, sin_theta, cos_theta, norm):
    if order == -6:
        return norm[6] * 10395 * sin_theta**6 * np.sin(6 * phi)
    elif order == -5:
        return norm[5] * 10395 * sin_theta**5 * cos_theta * np.sin(5 * phi)
    elif order == -4:
        return norm[4] * 472.5 * sin_theta**4 * (11 * cos_theta ** 2 - 1) * np.sin(4 * phi)
    elif order == -3:
        return norm[3] * 157.5 * sin_theta**3 * (11 * cos_theta ** 3 - 3 * cos_theta) * np.sin(3 * phi)
    elif order == -2:
        return norm[2] * 13.125 * sin_theta**2 * (33 * cos_theta ** 4 - 18 * cos_theta ** 2 + 1) * np.sin(2 * phi)
    elif order == -1:
        return norm[1] * 2.625 * sin_theta \
            * (33 * cos_theta**5 - 30 * cos_theta**3 + 5 * cos_theta) * np.sin(phi)
    elif order == 0:
        return norm[0] * 0.0625 * (231 * cos_theta ** 6 - 315 * cos_theta ** 4 + 105 * cos_theta ** 2 - 5)
    elif order == 1:
        return norm[1] * 2.625 * sin_theta \
            * (33 * cos_theta**5 - 30 * cos_theta**3 + 5 * cos_theta) * np.cos(phi)
    elif order == 2:
        return norm[2] * 13.125 * sin_theta**2 * (33 * cos_theta ** 4 - 18 * cos_theta ** 2 + 1) * np.cos(2 * phi)
    elif order == 3:
        return norm[3] * 157.5 * sin_theta**3 * (11 * cos_theta ** 3 - 3 * cos_theta) * np.cos(3 * phi)
    elif order == 4:
        return norm[4] * 472.5 * sin_theta**4 * (11 * cos_theta ** 2 - 1) * np.cos(4 * phi)
    elif order == 5:
        return norm[5] * 10395 * sin_theta**5 * cos_theta * np.cos(5 * phi)
    elif order == 6:
        return norm[6] * 10395 * sin_theta**6 * np.cos(6 * phi)


@njit(cache=True)
def sph_harm_degree_8(order, phi, sin_theta, cos_theta, norm):
    if order == -8:
        return norm[8] * 2027025 * sin_theta**8 * np.sin(8 * phi)
    elif order == -7:
        return norm[7] * 2027025 * sin_theta**7 * cos_theta * np.sin(7 * phi)
    elif order == -6:
        return norm[6] * 67567.5 * sin_theta**6 * (15 * cos_theta ** 2 - 1) * np.sin(6 * phi)
    elif order == -5:
        return norm[5] * 67567.5 * sin_theta**5 * (5 * cos_theta ** 3 - cos_theta) * np.sin(5 * phi)
    elif order == -4:
        return norm[4] * 1299.375 * sin_theta**4 * (65 * cos_theta ** 4 - 26 * cos_theta ** 2 + 1) * np.sin(4 * phi)
    elif order == -3:
        return norm[3] * 433.125 * sin_theta**3 \
            * (39 * cos_theta**5 - 26 * cos_theta**3 + 3 * cos_theta) * np.sin(3 * phi)
    elif order == -2:
        return norm[2] * 19.6875 * sin_theta**2 \
            * (143 * cos_theta**6 - 143 * cos_theta**4 + 33 * cos_theta**2 - 1) * np.sin(2 * phi)
    elif order == -1:
        return norm[1] * 0.5625 * sin_theta \
            * (715 * cos_theta**7 - 1001 * cos_theta**5 + 385 * cos_theta**3 - 35 * cos_theta) * np.sin(phi)
    elif order == 0:
        return norm[0] * 0.0078125 \
            * (6435 * cos_theta**8 - 12012 * cos_theta**6 + 6930 * cos_theta**4 - 1260 * cos_theta**2 + 35)
    elif order == 1:
        return norm[1] * 0.5625 * sin_theta \
            * (715 * cos_theta**7 - 1001 * cos_theta**5 + 385 * cos_theta**3 - 35 * cos_theta) * np.cos(phi)
    elif order == 2:
        return norm[2] * 19.6875 * sin_theta**2 \
            * (143 * cos_theta**6 - 143 * cos_theta**4 + 33 * cos_theta**2 - 1) * np.cos(2 * phi)
    elif order == 3:
        return norm[3] * 433.125 * sin_theta**3 \
            * (39 * cos_theta**5 - 26 * cos_theta**3 + 3 * cos_theta) * np.cos(3 * phi)
    elif order == 4:
        return norm[4] * 1299.375 * sin_theta**4 * (65 * cos_theta ** 4 - 26 * cos_theta ** 2 + 1) * np.cos(4 * phi)
    elif order == 5:
        return norm[5] * 67567.5 * sin_theta**5 * (5 * cos_theta ** 3 - cos_theta) * np.cos(5 * phi)
    elif order == 6:
        return norm[6] * 67567.5 * sin_theta**6 * (15 * cos_theta ** 2 - 1) * np.cos(6 * phi)
    elif order == 7:
        return norm[7] * 2027025 * sin_theta**7 * cos_theta * np.cos(7 * phi)
    elif order == 8:
        return norm[8] * 2027025 * sin_theta**8 * np.cos(8 * phi)


@njit(cache=True)
def sph_harm_degree_10(order, phi, sin_theta, cos_theta, norm):
    if order == -10:
        return norm[10] * 654729075 * sin_theta**10 * np.sin(10 * phi)
    elif order == -9:
        return norm[9] * 654729075 * sin_theta**9 * cos_theta * np.sin(9 * phi)
    elif order == -8:
        return norm[8] * 17229712.5 * sin_theta**8 * (19 * cos_theta ** 2 - 1) * np.sin(8 * phi)
    elif order == -7:
        return norm[7] * 5743237.5 * sin_theta**7 * (19 * cos_theta ** 3 - 3 * cos_theta) * np.sin(7 * phi)
    elif order == -6:
        return norm[6] * 84459.375 * sin_theta**6 \
            * (323 * cos_theta**4 - 102 * cos_theta**2 + 3) * np.sin(6 * phi)
    elif order == -5:
        return norm[5] * 16891.875 * sin_theta**5 \
            * (323 * cos_theta**5 - 170 * cos_theta**3 + 15 * cos_theta) * np.sin(5 * phi)
    elif order == -4:
        return norm[4] * 2815.3125 * sin_theta**4 \
            * (323 * cos_theta**6 - 255 * cos_theta**4 + 45 * cos_theta**2 - 1) * np.sin(4 * phi)
    elif order == -3:
        return norm[3] * 402.1875 * sin_theta**3 \
            * (323 * cos_theta**7 - 357 * cos_theta**5 + 105 * cos_theta**3 - 7 * cos_theta) * np.sin(3 * phi)
    elif order == -2:
        return norm[2] * 3.8671875 * sin_theta**2 \
            * (4199 * cos_theta**8 - 6188 * cos_theta**6 + 2730 * cos_theta**4 - 364 * cos_theta**2 + 7) \
            * np.sin(2 * phi)
    elif order == -1:
        return norm[1] * 0.4296875 * sin_theta \
            * (4199 * cos_theta**9 - 7956 * cos_theta**7 + 4914 * cos_theta**5 - 1092 * cos_theta**3 + 63 * cos_theta) \
            * np.sin(phi)
    elif order == 0:
        return norm[0] * 0.00390625 \
            * (46189 * cos_theta**10 - 109395 * cos_theta**8 + 90090 * cos_theta**6
               - 30030 * cos_theta**4 + 3465 * cos_theta**2 - 63)
    elif order == 1:
        return norm[1] * 0.4296875 * sin_theta \
            * (4199 * cos_theta**9 - 7956 * cos_theta**7 + 4914 * cos_theta**5
               - 1092 * cos_theta**3 + 63 * cos_theta) * np.cos(phi)
    elif order == 2:
        return norm[2] * 3.8671875 * sin_theta**2 \
            * (4199 * cos_theta**8 - 6188 * cos_theta**6 + 2730 * cos_theta**4 - 364 * cos_theta**2 + 7) \
            * np.cos(2 * phi)
    elif order == 3:
        return norm[3] * 402.1875 * sin_theta**3 \
            * (323 * cos_theta**7 - 357 * cos_theta**5 + 105 * cos_theta**3 - 7 * cos_theta) * np.cos(3 * phi)
    elif order == 4:
        return norm[4] * 2815.3125 * sin_theta**4 \
            * (323 * cos_theta**6 - 255 * cos_theta**4 + 45 * cos_theta**2 - 1) * np.cos(4 * phi)
    elif order == 5:
        return norm[5] * 16891.875 * sin_theta**5 \
            * (323 * cos_theta**5 - 170 * cos_theta**3 + 15 * cos_theta) * np.cos(5 * phi)
    elif order == 6:
        return norm[6] * 84459.375 * sin_theta**6 \
            * (323 * cos_theta**4 - 102 * cos_theta**2 + 3) * np.cos(6 * phi)
    elif order == 7:
        return norm[7] * 5743237.5 * sin_theta**7 * (19 * cos_theta ** 3 - 3 * cos_theta) * np.cos(7 * phi)
    elif order == 8:
        return norm[8] * 17229712.5 * sin_theta**8 * (19 * cos_theta ** 2 - 1) * np.cos(8 * phi)
    elif order == 9:
        return norm[9] * 654729075 * sin_theta**9 * cos_theta * np.cos(9 * phi)
    elif order == 10:
        return norm[10] * 654729075 * sin_theta**10 * np.cos(10 * phi)
