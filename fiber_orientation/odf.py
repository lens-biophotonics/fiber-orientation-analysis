import numpy as np
from numba import njit
from skimage.transform import resize


def compute_scaled_odf(odf_scale, vec_volume, iso_fiber_array, odf_patch_shape, degrees=6):                       
    """
    Iteratively generate 3D ODF maps at the desired spatial scale from basic slices
    of the fiber orientation vectors returned by the Frangi filtering stage.

    Parameters
    ----------
    odf_scale: int
        fiber ODF resolution (super-voxel side in [px])

    vec_volume: ndarray (shape=(Z,Y,X,3), dtype=float32)
        fiber orientation vectors

    iso_fiber_array: ndarray (shape=(Z,Y,X,3), dtype=uint8)
        isotropic fiber volume array

    odf_patch_shape: ndarray (shape=(Z,Y,X), dtype=int)
        3D shape of the output ODF data chunk (coefficients axis excluded)

    degrees: int
        degrees of the spherical harmonics series expansion

    Returns
    -------
    odf: ndarray (shape=(Z,Y,X, n_coeff), dtype=float32)
        array of spherical harmonics coefficients

    bg_mrtrix: ndarray (shape=(Z,Y,X), dtype=uint8)
        downsampled ODF background image (fiber channel)
    """
    # generate downsampled background for Mrtrix3 mrview
    if iso_fiber_array is None:
        bg_mrtrix = generate_odf_background(vec_volume, vxl_side=odf_scale)                                            
    else:
        bg_mrtrix = generate_odf_background(iso_fiber_array, vxl_side=odf_scale)                                            

    # compute ODF coefficients
    odf = estimate_odf_coeff(vec_volume, odf_patch_shape, vxl_side=odf_scale, degrees=degrees)                             

    return odf, bg_mrtrix


def generate_odf_background(bg_volume, vxl_side):
    """
    Generate the downsampled background image needed
    to visualize the 3D ODF map in Mrtrix3.

    Parameters
    ----------
    bg_volume: ndarray (shape=(Z,Y,X), dtype=uint8; or shape=(Z,Y,X,3), dtype=float32)                                      
        fiber image volume or vector volume
        to be used as the Mrtrix3 background image

    vxl_side: int
        side of the ODF super-voxel

    Returns
    -------
    bg_mrtrix: ndarray (shape=(Z,Y,X), dtype=uint8)
        downsampled ODF background (fiber channel)
    """
    # get shape of new downsampled array
    new_shape \
        = tuple(np.ceil(np.divide(bg_volume.shape[:3],
                                  vxl_side)).astype(int))

    # loop over z-slices, and resize them
    bg_mrtrix = np.zeros(new_shape, dtype=np.uint8)
    dims = bg_volume.ndim
    z_out = 0
    for z in range(0, bg_volume.shape[0], vxl_side):
        if dims == 3:
            tmp_slice = bg_volume[z, ...].copy()
        elif dims == 4:
            tmp_slice = 255.0*np.sum(np.abs(bg_volume[z, ...]), axis=-1)
            tmp_slice = np.where(tmp_slice <= 255.0, tmp_slice, 255.0)
            tmp_slice = tmp_slice.astype(np.uint8)
        bg_mrtrix[z_out, ...] = resize(tmp_slice,
                                       output_shape=new_shape[1:],
                                       anti_aliasing=True, preserve_range=True)
        z_out += 1

    return bg_mrtrix


def estimate_odf_coeff(vec_volume, odf_patch_shape, vxl_side, degrees, vxl_thr=0.5):                       
    """
    Estimate the spherical harmonics coefficients iterating over super-voxels
    of fiber orientation vectors.

    Parameters
    ----------
    vec_volume: ndarray (shape=(Z,Y,X,3), dtype=float)
        fiber orientation vectors

    odf_patch_shape: ndarray (shape=(Z,Y,X), dtype=int)
        3D shape of the output ODF data chunk (coefficients axis excluded)

    vxl_side: int
        side of the ODF super-voxel

    degrees: int
        degrees of the spherical harmonics series expansion

    vec_thr: float
        minimum relative threshold on the sliced voxel volume (default: 50%)        

    Returns
    -------
    odf_coeff: ndarray (shape=(Z,Y,X,ncoeff), dtype=float32)
        volumetric map of real-valued spherical harmonics coefficients
    """
    # initialize array of ODF coefficients
    ncoeff = get_sph_harm_ncoeff(degrees)
    odf_shape = tuple(list(odf_patch_shape) + [ncoeff])

    # initialize ODF array
    odf_coeff = np.zeros(odf_shape, dtype='float32')

    # compute spherical harmonics normalization factors (once)
    norm_factors = get_sph_harm_norm_factors(degrees)

    # impose a relative threshold on zero orientation vectors
    ref_vxl_size = min(vxl_side, vec_volume.shape[0]) * vxl_side**2
    for z in range(0, vec_volume.shape[0], vxl_side):
        zmax = z + vxl_side
        if zmax >= vec_volume.shape[0]:
            zmax = vec_volume.shape[0]

        for y in range(0, vec_volume.shape[1], vxl_side):
            ymax = y + vxl_side
            if ymax >= vec_volume.shape[1]:
                ymax = vec_volume.shape[1]

            for x in range(0, vec_volume.shape[2], vxl_side):
                xmax = x + vxl_side
                if xmax >= vec_volume.shape[2]:
                    xmax = vec_volume.shape[2]

                # slice vector voxel (skip boundary voxels)
                vec_vxl = vec_volume[z:zmax, y:ymax, x:xmax, :]
                sli_vxl_size = np.prod(vec_vxl.shape[:-1])
                if sli_vxl_size / ref_vxl_size > vxl_thr:
                    odf_coeff[z // vxl_side, y // vxl_side, x // vxl_side, :] \
                        = fiber_vectors_to_sph_harm(vec_vxl.ravel(), degrees, norm_factors)                                                        

    return odf_coeff


def fiber_vectors_to_sph_harm(vec_array, degrees, norm_factors):
    """
    Generate the real-valued symmetric spherical harmonics series expansion
    from the fiber orientation vectors returned by the Frangi filtering stage.

    Parameters
    ----------
    vec_array: ndarray (shape=(N,3), dtype=float)
        array of fiber orientation vectors
        (flattened super-voxel of shape=(Z,Y,X), i.e. N=Z*Y*X)

    degrees: int
        degrees of the spherical harmonics expansion

    norm_factors: ndarray (dtype: float)
        normalization factors

    Returns
    -------
    real_sph_harm: ndarray (shape=(ncoeff,), dtype=float)
        real-valued spherical harmonics coefficients
    """
    vec_array.shape = (-1, 3)
    ncoeff = get_sph_harm_ncoeff(degrees)

    norm = np.linalg.norm(vec_array, axis=-1)
    if np.sum(norm) < np.sqrt(vec_array.shape[0]):
        return np.zeros(ncoeff)

    phi, theta = compute_fiber_angles(vec_array, norm)

    real_sph_harm = fiber_angles_to_sph_harm(phi, theta, degrees, norm_factors, ncoeff)

    return real_sph_harm


@njit(cache=True)
def compute_fiber_angles(vec_array, norm):
    """
    Estimate the spherical coordinates (azimuth (φ) and polar (θ) angles)
    of the fiber orientation vectors returned by the Frangi filtering stage
    (all-zero background vectors are excluded).

    Parameters
    ----------
    vec_array: ndarray (shape=(N,3), dtype=float)
        array of fiber orientation vectors
        (flattened super-voxel of shape=(Z,Y,X), i.e. N=Z*Y*X)

    norm: ndarray (shape=(N,), dtype=float)
        2-norm of fiber orientation vectors

    Returns
    -------
    phi: ndarray (shape=(N,), dtype=float)
        fiber azimuth angle [rad]

    theta: ndarray (shape=(N,), dtype=float)
        fiber polar angle [rad]
    """
    vec_array = vec_array[norm > 0, :]
    phi = np.arctan2(vec_array[:, 1], vec_array[:, 2])
    theta = np.arccos(vec_array[:, 0] / norm[norm > 0])

    return phi, theta


@njit(cache=True)
def fiber_angles_to_sph_harm(phi, theta, degrees, norm_factors, ncoeff):
    """
    Generate the real-valued symmetric spherical harmonics series expansion
    from fiber azimuth (φ) and polar (θ) angles,
    i.e. the spherical coordinates of the fiber orientation vectors.

    Parameters
    ----------
    phi: ndarray (shape=(N,), dtype=float)
        fiber azimuth angles [rad]
        (flattened super-voxel of shape=(Z,Y,X), i.e. N=Z*Y*X)

    theta: ndarray (shape=(N,), dtype=float)
        fiber polar angle [rad]
        (flattened super-voxel of shape=(Z,Y,X), i.e. N=Z*Y*X)

    degrees: int
        degrees of the spherical harmonics expansion

    norm_factors: ndarray (dtype: float)
        normalization factors

    ncoeff: int
        number of spherical harmonics coefficients

    Returns
    -------
    real_sph_harm: ndarray (shape=(ncoeff,), dtype=float)
        real-valued spherical harmonics coefficients
    """
    real_sph_harm = np.zeros(ncoeff)
    i = 0
    for n in np.arange(0, degrees + 1, 2):
        for m in np.arange(-n, n + 1, 1):
            for j, (p, t) in enumerate(zip(phi, theta)):
                real_sph_harm[i] += compute_real_sph_harm(n, m, p, np.sin(t), np.cos(t), norm_factors)
            i += 1

    real_sph_harm = real_sph_harm / phi.size

    return real_sph_harm


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

    norm_factors: ndarray (dtype: float)
        normalization factors

    Returns
    -------
    spherical harmonics building the spherical harmonics series expansion
    """
    if degree == 0:
        return norm_factors[0, 0]
    elif degree == 2:
        return sph_harm_degree_2(order, phi, sin_theta, cos_theta, norm_factors[1, :])
    elif degree == 4:
        return sph_harm_degree_4(order, phi, sin_theta, cos_theta, norm_factors[2, :])                                           
    elif degree == 6:
        return sph_harm_degree_6(order, phi, sin_theta, cos_theta, norm_factors[3, :])                                           
    elif degree == 8:
        return sph_harm_degree_8(order, phi, sin_theta, cos_theta, norm_factors[4, :])                                           
    elif degree == 10:
        return sph_harm_degree_10(order, phi, sin_theta, cos_theta, norm_factors[5, :])
    else:
        raise(ValueError("\n  Invalid degree of the spherical harmonics series expansion!!!"))


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
    norm_factors_arr: ndarray (dtype: float)
        2D array of spherical harmonics normalization factors
    """
    norm_factors_arr = np.zeros(shape=(degrees + 1, 2 * degrees + 1))
    for n in np.arange(0, degrees + 1, 2):
        for m in np.arange(0, n + 1, 1):
            norm_factors_arr[n, m] = norm_factor(n, m)

    norm_factors_arr = norm_factors_arr[::2]

    return norm_factors_arr


factorial_lut = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype=np.double)


@njit(cache=True)
def factorial(n):
    if n > 20:
        raise ValueError
    return factorial_lut[n]


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
    normalization factor
    """
    if m == 0:
        return np.sqrt((2 * n + 1)/(4 * np.pi))
    else:
        return (-1)**m * np.sqrt(2) \
            * np.sqrt(((2 * n + 1)/(4 * np.pi) *
                       (factorial(n - np.abs(m)) / factorial(n + np.abs(m)))))


@njit(cache=True)
def sph_harm_degree_2(order, phi, sin_theta, cos_theta, norm_factor):                                
    if order == -2:
        return norm_factor[2] * 3 * sin_theta**2 * np.sin(2 * phi)
    elif order == -1:
        return norm_factor[1] * 3 * sin_theta * cos_theta * np.sin(phi)
    elif order == 0:
        return norm_factor[0] * 0.5 * (3 * cos_theta**2 - 1)
    elif order == 1:
        return norm_factor[1] * 3 * sin_theta * cos_theta * np.cos(phi)
    elif order == 2:
        return norm_factor[2] * 3 * sin_theta**2 * np.cos(2 * phi)


@njit(cache=True)
def sph_harm_degree_4(order, phi, sin_theta, cos_theta, norm_factor):                                
    if order == -4:
        return norm_factor[4] * 105 * sin_theta**4 * np.sin(4 * phi)
    elif order == -3:
        return norm_factor[3] * 105 * sin_theta**3 * cos_theta * np.sin(3 * phi)
    elif order == -2:
        return norm_factor[2] * 7.5 * sin_theta**2 * (7 * cos_theta**2 - 1) * np.sin(2 * phi)            
    elif order == -1:
        return norm_factor[1] * 2.5 * sin_theta*(7 * cos_theta**3 - 3 * cos_theta) * np.sin(phi)            
    elif order == 0:
        return norm_factor[0] * 0.125 * (35 * cos_theta**4 - 30 * cos_theta**2 + 3)
    elif order == 1:
        return norm_factor[1] * 2.5 * sin_theta * (7 * cos_theta**3 - 3 * cos_theta) * np.cos(phi)            
    elif order == 2:
        return norm_factor[2] * 7.5 * sin_theta**2 * (7 * cos_theta**2 - 1) * np.cos(2 * phi)            
    elif order == 3:
        return norm_factor[3] * 105 * sin_theta**3 * cos_theta*np.cos(3 * phi)
    elif order == 4:
        return norm_factor[4] * 105 * sin_theta**4 * np.cos(4 * phi)


@njit(cache=True)
def sph_harm_degree_6(order, phi, sin_theta, cos_theta, norm_factor):                                
    if order == -6:
        return norm_factor[6] * 10395 * sin_theta**6 * np.sin(6 * phi)
    elif order == -5:
        return norm_factor[5] * 10395 * sin_theta**5 * cos_theta * np.sin(5 * phi)
    elif order == -4:
        return norm_factor[4] * 472.5 * sin_theta**4 * (11 * cos_theta**2 - 1) * np.sin(4 * phi)
    elif order == -3:
        return norm_factor[3] * 157.5 * sin_theta**3 * (11 * cos_theta**3 - 3 * cos_theta) * np.sin(3 * phi)
    elif order == -2:
        return norm_factor[2] * 13.125 * sin_theta**2 * (33 * cos_theta**4 - 18 * cos_theta**2 + 1) * np.sin(2 * phi)
    elif order == -1:
        return norm_factor[1] * 2.625 * sin_theta \
            * (33 * cos_theta**5 - 30 * cos_theta**3 + 5 * cos_theta) * np.sin(phi)
    elif order == 0:
        return norm_factor[0] * 0.0625 * (231 * cos_theta**6 - 315 * cos_theta**4 + 105 * cos_theta**2 - 5)
    elif order == 1:
        return norm_factor[1] * 2.625 * sin_theta \
            * (33 * cos_theta**5 - 30 * cos_theta**3 + 5 * cos_theta) * np.cos(phi)
    elif order == 2:
        return norm_factor[2] * 13.125 * sin_theta**2 * (33 * cos_theta**4 - 18 * cos_theta**2 + 1) * np.cos(2 * phi)
    elif order == 3:
        return norm_factor[3] * 157.5 * sin_theta**3 * (11 * cos_theta**3 - 3 * cos_theta) * np.cos(3 * phi)
    elif order == 4:
        return norm_factor[4] * 472.5 * sin_theta**4 * (11 * cos_theta**2 - 1) * np.cos(4 * phi)
    elif order == 5:
        return norm_factor[5] * 10395 * sin_theta**5 * cos_theta * np.cos(5 * phi)
    elif order == 6:
        return norm_factor[6] * 10395 * sin_theta**6 * np.cos(6 * phi)


@njit(cache=True)
def sph_harm_degree_8(order, phi, sin_theta, cos_theta, norm_factor):                                
    if order == -8:
        return norm_factor[8] * 2027025 * sin_theta**8 * np.sin(8 * phi)
    elif order == -7:
        return norm_factor[7] * 2027025 * sin_theta**7 * cos_theta * np.sin(7 * phi)
    elif order == -6:
        return norm_factor[6] * 67567.5 * sin_theta**6 * (15 * cos_theta**2 - 1) * np.sin(6 * phi)
    elif order == -5:
        return norm_factor[5] * 67567.5 * sin_theta**5 * (5 * cos_theta**3 - cos_theta) * np.sin(5 * phi)            
    elif order == -4:
        return norm_factor[4] * 1299.375 * sin_theta**4 * (65 * cos_theta**4 - 26 * cos_theta**2 + 1) * np.sin(4 * phi)            
    elif order == -3:
        return norm_factor[3] * 433.125 * sin_theta**3 \
            * (39 * cos_theta**5 - 26 * cos_theta**3 + 3 * cos_theta) * np.sin(3 * phi)
    elif order == -2:
        return norm_factor[2] * 19.6875 * sin_theta**2 \
            * (143 * cos_theta**6 - 143 * cos_theta**4 + 33 * cos_theta**2 - 1) * np.sin(2 * phi)            
    elif order == -1:
        return norm_factor[1] * 0.5625 * sin_theta \
            * (715 * cos_theta**7 - 1001 * cos_theta**5 + 385 * cos_theta**3 - 35 * cos_theta) * np.sin(phi)               
    elif order == 0:
        return norm_factor[0] * 0.0078125 \
            * (6435 * cos_theta**8 - 12012 * cos_theta**6 + 6930 * cos_theta**4 - 1260 * cos_theta**2 + 35)               
    elif order == 1:
        return norm_factor[1] * 0.5625 * sin_theta \
            * (715 * cos_theta**7 - 1001 * cos_theta**5 + 385 * cos_theta**3 - 35 * cos_theta) * np.cos(phi)               
    elif order == 2:
        return norm_factor[2] * 19.6875 * sin_theta**2 \
            * (143 * cos_theta**6 - 143 * cos_theta**4 + 33 * cos_theta**2 - 1) * np.cos(2 * phi)            
    elif order == 3:
        return norm_factor[3] * 433.125 * sin_theta**3 \
            * (39 * cos_theta**5 - 26 * cos_theta**3 + 3 * cos_theta) * np.cos(3 * phi)
    elif order == 4:
        return norm_factor[4] * 1299.375 * sin_theta**4 * (65 * cos_theta**4 - 26 * cos_theta**2 + 1) * np.cos(4 * phi)            
    elif order == 5:
        return norm_factor[5] * 67567.5 * sin_theta**5 * (5 * cos_theta**3 - cos_theta) * np.cos(5 * phi)
    elif order == 6:
        return norm_factor[6] * 67567.5 * sin_theta**6 * (15 * cos_theta**2 - 1) * np.cos(6 * phi)
    elif order == 7:
        return norm_factor[7] * 2027025 * sin_theta**7 * cos_theta * np.cos(7 * phi)
    elif order == 8:
        return norm_factor[8] * 2027025 * sin_theta**8 * np.cos(8 * phi)


@njit(cache=True)
def sph_harm_degree_10(order, phi, sin_theta, cos_theta,  norm_factor):                                
    if order == -10:
        return norm_factor[10] * 654729075 * sin_theta**10 * np.sin(10 * phi)
    elif order == -9:
        return norm_factor[9] * 654729075 * sin_theta**9 * cos_theta * np.sin(9 * phi)
    elif order == -8:
        return norm_factor[8] * 17229712.5 * sin_theta**8 * (19 * cos_theta**2 - 1) * np.sin(8 * phi)            
    elif order == -7:
        return norm_factor[7] * 5743237.5 * sin_theta**7 * (19 * cos_theta**3 - 3 * cos_theta) * np.sin(7 * phi)            
    elif order == -6:
        return norm_factor[6] * 84459.375 * sin_theta**6 \
            * (323 * cos_theta**4 - 102 * cos_theta**2 + 3) * np.sin(6 * phi)
    elif order == -5:
        return norm_factor[5] * 16891.875 * sin_theta**5 \
            * (323 * cos_theta**5 - 170 * cos_theta**3 + 15 * cos_theta) * np.sin(5 * phi)            
    elif order == -4:
        return norm_factor[4] * 2815.3125 * sin_theta**4 \
            * (323 * cos_theta**6 - 255 * cos_theta**4 + 45 * cos_theta**2 - 1) * np.sin(4 * phi)            
    elif order == -3:
        return norm_factor[3] * 402.1875 * sin_theta**3 \
            * (323 * cos_theta**7 - 357 * cos_theta**5 + 105 * cos_theta**3 - 7 * cos_theta) * np.sin(3 * phi)               
    elif order == -2:
        return norm_factor[2] * 3.8671875 * sin_theta**2 \
            * (4199 * cos_theta**8 - 6188 * cos_theta**6 + 2730 * cos_theta**4 - 364 * cos_theta**2 + 7) \
            * np.sin(2 * phi)
    elif order == -1:
        return norm_factor[1] * 0.4296875 * sin_theta \
            * (4199 * cos_theta**9 - 7956 * cos_theta**7 + 4914 * cos_theta**5 - 1092 * cos_theta**3 + 63 * cos_theta) \
            * np.sin(phi)
    elif order == 0:
        return norm_factor[0] * 0.00390625 \
            * (46189 * cos_theta**10 - 109395 * cos_theta**8 + 90090 * cos_theta**6 
               - 30030*cos_theta**4 + 3465 * cos_theta**2 - 63)               
    elif order == 1:
        return norm_factor[1] * 0.4296875 * sin_theta \
            * (4199 * cos_theta**9 - 7956 * cos_theta**7 + 4914 * cos_theta**5
               - 1092 * cos_theta**3 + 63 * cos_theta) * np.cos(phi)
    elif order == 2:
        return norm_factor[2] * 3.8671875 * sin_theta**2 \
            * (4199 * cos_theta**8 - 6188 * cos_theta**6 + 2730 * cos_theta**4 - 364 * cos_theta**2 + 7) \
            * np.cos(2 * phi)            
    elif order == 3:
        return norm_factor[3] * 402.1875 * sin_theta**3 \
            * (323 * cos_theta**7 - 357 * cos_theta**5 + 105 * cos_theta**3 - 7 * cos_theta) * np.cos(3 * phi)               
    elif order == 4:
        return norm_factor[4] * 2815.3125 * sin_theta**4 \
            * (323 * cos_theta**6 - 255 * cos_theta**4 + 45 * cos_theta**2 - 1) * np.cos(4 * phi)            
    elif order == 5:
        return norm_factor[5] * 16891.875 * sin_theta**5 \
            * (323 * cos_theta**5 - 170 * cos_theta**3 + 15 * cos_theta) * np.cos(5 * phi)            
    elif order == 6:
        return norm_factor[6] * 84459.375 * sin_theta**6 \
            * (323 * cos_theta**4 - 102 * cos_theta**2 + 3) * np.cos(6 * phi)            
    elif order == 7:
        return norm_factor[7] * 5743237.5 * sin_theta**7 * (19 * cos_theta**3 - 3 * cos_theta) * np.cos(7 * phi)            
    elif order == 8:
        return norm_factor[8] * 17229712.5 * sin_theta**8 * (19*cos_theta**2 - 1) * np.cos(8 * phi)            
    elif order == 9:
        return norm_factor[9] * 654729075 * sin_theta**9 * cos_theta * np.cos(9 * phi)
    elif order == 10:
        return norm_factor[10] * 654729075 * sin_theta**10 * np.cos(10 * phi)
