import numpy as np
from numba import njit
from skimage.transform import resize


def compute_scaled_odf(odf_scale, vec_array, iso_fiber_array, odf_patch_shape,
                       orders=6):
    """
    Iteratively generate 3D ODF maps at desired scale from basic slices
    of the fiber orientation vectors returned by the Frangi filtering stage.

    Parameters
    ----------
    odf_scale: int
        fiber ODF resolution (super-voxel side in [px])

    vec_array: ndarray (shape=(Z,Y,X,3), dtype=float32)
        orientation vector array

    iso_fiber_array: ndarray (shape=(Z,Y,X,3), dtype=uint8)
        isotropic fiber volume array

    odf_patch_shape: ndarray (shape=(Z,Y,X), dtype=int)
        3D shape of the output ODF data chunk (coefficients axis excluded)

    orders: int
        orders of the spherical harmonics series expansion

    Returns
    -------
    odf: ndarray (shape=(Z,Y,X, n_coeff), dtype=float32)
        array of spherical harmonics coefficients

    bg_mrtrix: ndarray (shape=(Z,Y,X), dtype=uint8)
        downsampled ODF background image (fiber channel)
    """
    # generate downsampled background for Mrtrix3 mrview
    if iso_fiber_array is None:
        bg_mrtrix = generate_odf_background(vec_array,
                                            vxl_side=odf_scale)
    else:
        bg_mrtrix = generate_odf_background(iso_fiber_array,
                                            vxl_side=odf_scale)

    # compute ODF coefficients
    odf = estimate_odf_coeff(vec_array, odf_patch_shape,
                             vxl_side=odf_scale, orders=orders)

    return odf, bg_mrtrix


def generate_odf_background(bg_volume, vxl_side):
    """
    Generate the downsampled background image needed
    to visualize the 3D ODF map in Mrtrix3.

    Parameters
    ----------
    bg_volume: ndarray (shape=(Z,Y,X), dtype=uint8;
                                      shape=(Z,Y,X,3), dtype=float32)
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


def estimate_odf_coeff(vec_volume, odf_patch_shape, vxl_side, orders,
                       vxl_thr=0.5):
    """
    Estimate the spherical harmonics coefficients iterating over super-voxels
    of fiber orientation vectors.

    Parameters
    ----------
    vec_volume: ndarray (shape=(Z,Y,X,3), dtype=float)
        dominant eigenvector volume

    odf_patch_shape: ndarray (shape=(Z,Y,X), dtype=int)
        3D shape of the output ODF data chunk (coefficients axis excluded)

    vxl_side: int
        side of the ODF super-voxel

    orders: int
        orders of the spherical harmonics series expansion

    vec_thr: float
        minimum relative threshold on the sliced voxel volume
        (default: 50%)

    Returns
    -------
    odf_coeff: ndarray (shape=(Z,Y,X,n_coeff), dtype=float32)
        spherical harmonics coefficients
    """
    # initialize array of ODF coefficients
    num_coeff = get_sh_coef_num(orders)
    odf_shape = tuple(list(odf_patch_shape) + [num_coeff])

    # initialize ODF array
    odf_coeff = np.zeros(odf_shape, dtype='float32')

    # compute spherical harmonics normalization factors (once)
    norm_factors = get_sh_norm_factors(orders)

    # impose a relative threshold on zero orientation vectors
    ref_vxl_size = min(vxl_side, vec_volume.shape[0])*vxl_side**2
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

                # slice vector voxel
                # (skip boundary voxels)
                vec_vxl = vec_volume[z:zmax, y:ymax, x:xmax, :]
                sli_vxl_size = np.prod(vec_vxl.shape[:-1])
                if sli_vxl_size / ref_vxl_size > vxl_thr:
                    odf_coeff[z // vxl_side, y // vxl_side, x // vxl_side, :] \
                        = vectors_to_spherical_harmonics(vec_vxl.ravel(),
                                                         orders,
                                                         norm_factors)

    return odf_coeff


def vectors_to_spherical_harmonics(vec_volume, orders, norm_factors):
    """
    Generate the real-valued symmetric spherical harmonics series expansion
    from the fiber orientation vectors returned by the Frangi filtering stage.

    Parameters
    ----------
    vec_volume: ndarray (shape=(Z,Y,X,3), dtype=float)
        fiber orientation vectors

    orders: int
        orders of the spherical harmonics expansion

    norm_factors: ndarray (dtype: float)
        normalization factors

    Returns
    -------
    real_sh: ndarray (shape=(Z,Y,X,n_coeff), dtype=float)
        spherical harmonics coefficients
    """
    vec_volume.shape = (-1, 3)
    n_coeff = get_sh_coef_num(orders)

    norm = np.linalg.norm(vec_volume, axis=-1)
    if np.sum(norm) < np.sqrt(vec_volume.shape[0]):
        return np.zeros(n_coeff)

    phi, theta = compute_fiber_angles(vec_volume, norm)

    return angles_to_spherical_harmonics(phi, theta, orders, norm_factors)


@njit(cache=True)
def compute_fiber_angles(vec_volume, norm):
    """
    Estimate the spherical coordinates (azimuth (φ) and polar (θ) angles)
    of the fiber orientation vectors returned by the Frangi filtering stage
    (all-zero background vectors are excluded).

    Parameters
    ----------
    vec_volume: ndarray (shape=(Z,Y,X,3), dtype=float)
        fiber orientation vectors

    norm: ndarray (shape=(Z,Y,X), dtype=float)
        2-norm of fiber orientation vectors

    Returns
    -------
    phi: ndarray (shape=(Z,Y,X), dtype=float)
        fiber azimuth angle

    theta: ndarray (shape=(Z,Y,X), dtype=float)
        fiber polar angle
    """
    vec_volume = vec_volume[norm > 0, :]
    phi = np.arctan2(vec_volume[:, 1], vec_volume[:, 2])
    theta = np.arccos(vec_volume[:, 0] / norm[norm > 0])

    return phi, theta


@njit(cache=True)
def angles_to_spherical_harmonics(phi, theta, orders, norm_factors):
    """
    Generate the real-valued symmetric spherical harmonics series expansion
    from fiber azimuth (φ) and polar (θ) angles,
    i.e. the spherical coordinates of the fiber orientation vectors.

    Parameters
    ----------
    phi: ndarray (shape=(Z,Y,X), dtype=float)
        fiber azimuth angle

    theta: ndarray (shape=(Z,Y,X), dtype=float)
        fiber polar angle

    orders: int
        orders of the spherical harmonics expansion

    norm_factors: ndarray (dtype: float)
        normalization factors

    Returns
    -------
    real_sh: ndarray (shape=(Z,Y,X,n_coeff), dtype=float)
        spherical harmonics coefficients
    """
    n_coeff = get_sh_coef_num(orders)
    real_sh = np.zeros(n_coeff)
    i = 0
    for o in np.arange(0, orders + 1, 2):
        for m in np.arange(-o, o + 1, 1):
            for j, (p, t) in enumerate(zip(phi, theta)):
                real_sh[i] += spherical_harmonics(o, m, p,
                                                  np.sin(t), np.cos(t),
                                                  norm_factors)
            i += 1

    return real_sh / phi.size


@njit(cache=True)
def spherical_harmonics(order, phase_factor, phi, sin_theta, cos_theta,
                        norm_factors):
    """
    Estimate the coefficients of the real spherical harmonics series expansion
    as described by Alimi et al. (Medical Image Analysis, 2020).

    Parameters
    ----------
    order: int
        order index of the spherical harmonics expansion

    phase_factor: int
        phase factor index of the spherical harmonics expansion

    phi: float
        azimuth angle

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
    if order == 0:
        return norm_factors[0, 0]
    elif order == 2:
        return spherical_harmonics_order_2(phase_factor, phi,
                                           sin_theta, cos_theta,
                                           norm_factors[1, :])
    elif order == 4:
        return spherical_harmonics_order_4(phase_factor, phi,
                                           sin_theta, cos_theta,
                                           norm_factors[2, :])
    elif order == 6:
        return spherical_harmonics_order_6(phase_factor, phi,
                                           sin_theta, cos_theta,
                                           norm_factors[3, :])
    elif order == 8:
        return spherical_harmonics_order_8(phase_factor, phi,
                                           sin_theta, cos_theta,
                                           norm_factors[4, :])
    elif order == 10:
        return spherical_harmonics_order_10(phase_factor, phi,
                                            sin_theta, cos_theta,
                                            norm_factors[5, :])
    else:
        raise(ValueError("\n  Invalid spherical harmonics expansion index!!!"))


@njit(cache=True)
def get_sh_coef_num(orders):
    """
    Get the number of coefficients of the real spherical harmonics series
    expansion.

    Parameters
    ----------
    orders: int
        orders of the spherical harmonics series expansion

    Returns
    -------
    n_coef: int
        number of coefficients
    """
    n_coef = (2 * (orders // 2) + 1) * ((orders // 2) + 1)
    return n_coef


@njit(cache=True)
def get_sh_norm_factors(orders):
    """
    Estimate the normalization factors of the real spherical harmonics series
    expansion.

    Parameters
    ----------
    orders: int
        orders of the spherical harmonics series expansion

    Returns
    -------
    norm_factors_arr: ndarray (dtype: float)
        2D array of spherical harmonics normalization factors
    """
    norm_factors_arr = np.zeros(shape=(orders + 1, 2*orders + 1))
    for o in np.arange(0, orders + 1, 2):
        for m in np.arange(0, o + 1, 1):
            norm_factors_arr[o, m] = norm_factor(o, m)

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
def norm_factor(o, m):
    """
    Compute the normalization factor of the term of order o and phase factor m
    of the real spherical harmonics series expansion.

    Parameters
    ----------
    o: int
        order index

    m: int
        phase factor index

    Returns
    -------
    normalization factor
    """
    if m == 0:
        return np.sqrt((2*o + 1)/(4*np.pi))
    else:
        return (-1)**m * np.sqrt(2) \
            * np.sqrt(((2*o + 1)/(4*np.pi) *
                       (factorial(o - np.abs(m))/factorial(o + np.abs(m)))))


@njit(cache=True)
def spherical_harmonics_order_2(phase_factor, phi, sin_theta, cos_theta,
                                norm_factor):
    if phase_factor == -2:
        return norm_factor[2]*3.0*sin_theta**2*np.sin(2*phi)
    elif phase_factor == -1:
        return norm_factor[1]*3.0*sin_theta*cos_theta*np.sin(phi)
    elif phase_factor == 0:
        return norm_factor[0]*0.5*(3*cos_theta**2 - 1)
    elif phase_factor == 1:
        return norm_factor[1]*3.0*sin_theta*cos_theta*np.cos(phi)
    elif phase_factor == 2:
        return norm_factor[2]*3.0*sin_theta**2*np.cos(2*phi)
    else:
        raise(ValueError("\n  Invalid spherical harmonics expansion index!!!"))


@njit(cache=True)
def spherical_harmonics_order_4(phase_factor, phi, sin_theta, cos_theta,
                                norm_factor):
    if phase_factor == -4:
        return norm_factor[4]*105.0*sin_theta**4*np.sin(4*phi)
    elif phase_factor == -3:
        return norm_factor[3]*105.0*sin_theta**3*cos_theta*np.sin(3*phi)
    elif phase_factor == -2:
        return norm_factor[2]*7.5*sin_theta**2*(7*cos_theta**2 - 1) \
            * np.sin(2*phi)
    elif phase_factor == -1:
        return norm_factor[1]*2.5*sin_theta*(7*cos_theta**3 - 3*cos_theta) \
            * np.sin(phi)
    elif phase_factor == 0:
        return norm_factor[0]*0.125*(35*cos_theta**4 - 30*cos_theta**2+3)
    elif phase_factor == 1:
        return norm_factor[1]*2.5*sin_theta*(7*cos_theta**3 - 3*cos_theta) \
            * np.cos(phi)
    elif phase_factor == 2:
        return norm_factor[2]*7.5*sin_theta**2*(7*cos_theta**2 - 1) \
            * np.cos(2*phi)
    elif phase_factor == 3:
        return norm_factor[3]*105.0*sin_theta**3*cos_theta*np.cos(3*phi)
    elif phase_factor == 4:
        return norm_factor[4]*105.0*sin_theta**4*np.cos(4*phi)
    else:
        raise(ValueError("\n  Invalid spherical harmonics expansion index!!!"))


@njit(cache=True)
def spherical_harmonics_order_6(phase_factor, phi, sin_theta, cos_theta,
                                norm_factor):
    if phase_factor == -6:
        return norm_factor[6]*10395.0*sin_theta**6*np.sin(6*phi)
    elif phase_factor == -5:
        return norm_factor[5]*10395.0*sin_theta**5*cos_theta*np.sin(5*phi)
    elif phase_factor == -4:
        return norm_factor[4]*472.5*sin_theta**4*(11*cos_theta**2 - 1) \
            * np.sin(4*phi)
    elif phase_factor == -3:
        return norm_factor[3]*157.5*sin_theta**3 \
            * (11*cos_theta**3 - 3*cos_theta)*np.sin(3*phi)
    elif phase_factor == -2:
        return norm_factor[2]*13.125*sin_theta**2 \
            * (33*cos_theta**4 - 18*cos_theta**2 + 1)*np.sin(2*phi)
    elif phase_factor == -1:
        return norm_factor[1]*2.625*sin_theta \
            * (33*cos_theta**5 - 30*cos_theta**3 + 5*cos_theta)*np.sin(phi)
    elif phase_factor == 0:
        return norm_factor[0]*0.0625 \
            * (231*cos_theta**6 - 315*cos_theta**4 + 105*cos_theta**2-5)
    elif phase_factor == 1:
        return norm_factor[1]*2.625*sin_theta \
            * (33*cos_theta**5 - 30*cos_theta**3 + 5*cos_theta)*np.cos(phi)
    elif phase_factor == 2:
        return norm_factor[2]*13.125*sin_theta**2 \
            * (33*cos_theta**4 - 18*cos_theta**2 + 1)*np.cos(2*phi)
    elif phase_factor == 3:
        return norm_factor[3]*157.5*sin_theta**3 \
            * (11*cos_theta**3 - 3*cos_theta)*np.cos(3*phi)
    elif phase_factor == 4:
        return norm_factor[4]*472.5*sin_theta**4*(11*cos_theta**2 - 1) \
            * np.cos(4*phi)
    elif phase_factor == 5:
        return norm_factor[5]*10395.0*sin_theta**5*cos_theta*np.cos(5*phi)
    elif phase_factor == 6:
        return norm_factor[6]*10395.0*sin_theta**6*np.cos(6*phi)
    else:
        raise(ValueError("\n  Invalid spherical harmonics expansion index!!!"))


@njit(cache=True)
def spherical_harmonics_order_8(phase_factor, phi, sin_theta, cos_theta,
                                norm_factor):
    if phase_factor == -8:
        return norm_factor[8]*2027025.0*sin_theta**8*np.sin(8*phi)
    elif phase_factor == -7:
        return norm_factor[7]*2027025.0*sin_theta**7*cos_theta*np.sin(7*phi)
    elif phase_factor == -6:
        return norm_factor[6]*67567.5*sin_theta**6*(15*cos_theta**2 - 1) \
            * np.sin(6*phi)
    elif phase_factor == -5:
        return norm_factor[5]*67567.5*sin_theta**5 \
            * (5*cos_theta**3 - cos_theta)*np.sin(5*phi)
    elif phase_factor == -4:
        return norm_factor[4]*1299.375*sin_theta**4 \
            * (65*cos_theta**4 - 26*cos_theta**2+1)*np.sin(4*phi)
    elif phase_factor == -3:
        return norm_factor[3]*433.125*sin_theta**3 \
            * (39*cos_theta**5 - 26*cos_theta**3 + 3*cos_theta)*np.sin(3*phi)
    elif phase_factor == -2:
        return norm_factor[2]*19.6875*sin_theta**2 \
            * (143*cos_theta**6 - 143*cos_theta**4 + 33*cos_theta**2 - 1) \
            * np.sin(2*phi)
    elif phase_factor == -1:
        return norm_factor[1]*0.5625*sin_theta \
            * (715*cos_theta**7 - 1001*cos_theta**5 + 385*cos_theta**3
               - 35*cos_theta) \
            * np.sin(phi)
    elif phase_factor == 0:
        return norm_factor[0]*0.0078125 \
            * (6435*cos_theta**8 - 12012*cos_theta**6 + 6930*cos_theta**4
               - 1260*cos_theta**2 + 35)
    elif phase_factor == 1:
        return norm_factor[1]*0.5625*sin_theta \
            * (715*cos_theta**7 - 1001*cos_theta**5 + 385*cos_theta**3
               - 35*cos_theta)*np.cos(phi)
    elif phase_factor == 2:
        return norm_factor[2]*19.6875*sin_theta**2 \
            * (143*cos_theta**6 - 143*cos_theta**4 + 33*cos_theta**2 - 1) \
            * np.cos(2*phi)
    elif phase_factor == 3:
        return norm_factor[3]*433.125*sin_theta**3 \
            * (39*cos_theta**5 - 26*cos_theta**3 + 3*cos_theta)*np.cos(3*phi)
    elif phase_factor == 4:
        return norm_factor[4]*1299.375*sin_theta**4 \
            * (65*cos_theta**4 - 26*cos_theta**2 + 1)*np.cos(4*phi)
    elif phase_factor == 5:
        return norm_factor[5]*67567.5*sin_theta**5 \
            * (5*cos_theta**3 - cos_theta)*np.cos(5*phi)
    elif phase_factor == 6:
        return norm_factor[6]*67567.5*sin_theta**6*(15*cos_theta**2 - 1) \
            * np.cos(6*phi)
    elif phase_factor == 7:
        return norm_factor[7]*2027025.0*sin_theta**7 \
            * cos_theta*np.cos(7*phi)
    elif phase_factor == 8:
        return norm_factor[8]*2027025.0*sin_theta**8*np.cos(8*phi)
    else:
        raise(ValueError("\n  Invalid spherical harmonics expansion index!!!"))


@njit(cache=True)
def spherical_harmonics_order_10(phase_factor, phi, sin_theta, cos_theta,
                                 norm_factor):
    if phase_factor == -10:
        return norm_factor[10]*654729075.0*sin_theta**10*np.sin(10*phi)
    elif phase_factor == -9:
        return norm_factor[9]*654729075.0*sin_theta**9*cos_theta*np.sin(9*phi)
    elif phase_factor == -8:
        return norm_factor[8]*17229712.5*sin_theta**8*(19*cos_theta**2 - 1) \
            * np.sin(8*phi)
    elif phase_factor == -7:
        return norm_factor[7]*5743237.5*sin_theta**7 \
            * (19*cos_theta**3 - 3*cos_theta)*np.sin(7*phi)
    elif phase_factor == -6:
        return norm_factor[6]*84459.375*sin_theta**6 \
            * (323*cos_theta**4 - 102*cos_theta**2 + 3)*np.sin(6*phi)
    elif phase_factor == -5:
        return norm_factor[5]*16891.875*sin_theta**5 \
            * (323*cos_theta**5 - 170*cos_theta**3 + 15*cos_theta) \
            * np.sin(5*phi)
    elif phase_factor == -4:
        return norm_factor[4]*2815.3125*sin_theta**4 \
            * (323*cos_theta**6 - 255*cos_theta**4 + 45*cos_theta**2 - 1) \
            * np.sin(4*phi)
    elif phase_factor == -3:
        return norm_factor[3]*402.1875*sin_theta**3 \
            * (323*cos_theta**7 - 357*cos_theta**5 + 105*cos_theta**3
               - 7*cos_theta) * np.sin(3*phi)
    elif phase_factor == -2:
        return norm_factor[2]*3.8671875*sin_theta**2 \
            * (4199*cos_theta**8 - 6188*cos_theta**6 + 2730*cos_theta**4 - 364
               * cos_theta**2 + 7)*np.sin(2*phi)
    elif phase_factor == -1:
        return norm_factor[1]*0.4296875*sin_theta \
            * (4199*cos_theta**9 - 7956*cos_theta**7 + 4914*cos_theta**5
               - 1092*cos_theta**3 + 63*cos_theta)*np.sin(phi)
    elif phase_factor == 0:
        return norm_factor[0]*0.00390625 \
            * (46189*cos_theta**10 - 109395*cos_theta**8 + 90090*cos_theta**6 -
               30030*cos_theta**4 + 3465*cos_theta**2 - 63)
    elif phase_factor == 1:
        return norm_factor[1]*0.4296875*sin_theta \
            * (4199*cos_theta**9 - 7956*cos_theta**7 + 4914*cos_theta**5
               - 1092*cos_theta**3 + 63*cos_theta)*np.cos(phi)
    elif phase_factor == 2:
        return norm_factor[2]*3.8671875*sin_theta**2 \
            * (4199*cos_theta**8 - 6188*cos_theta**6 + 2730*cos_theta**4
               - 364*cos_theta**2 + 7)*np.cos(2*phi)
    elif phase_factor == 3:
        return norm_factor[3]*402.1875*sin_theta**3 \
            * (323*cos_theta**7 - 357*cos_theta**5 + 105*cos_theta**3
               - 7*cos_theta)*np.cos(3*phi)
    elif phase_factor == 4:
        return norm_factor[4]*2815.3125*sin_theta**4 \
            * (323*cos_theta**6 - 255*cos_theta**4 + 45*cos_theta**2 - 1) \
            * np.cos(4*phi)
    elif phase_factor == 5:
        return norm_factor[5]*16891.875*sin_theta**5 \
            * (323*cos_theta**5 - 170*cos_theta**3 + 15*cos_theta) \
            * np.cos(5*phi)
    elif phase_factor == 6:
        return norm_factor[6]*84459.375*sin_theta**6 \
            * (323*cos_theta**4 - 102*cos_theta**2 + 3)*np.cos(6*phi)
    elif phase_factor == 7:
        return norm_factor[7]*5743237.5*sin_theta**7 \
            * (19*cos_theta**3 - 3*cos_theta)*np.cos(7*phi)
    elif phase_factor == 8:
        return norm_factor[8]*17229712.5*sin_theta**8*(19*cos_theta**2 - 1) \
            * np.cos(8*phi)
    elif phase_factor == 9:
        return norm_factor[9]*654729075.0*sin_theta**9*cos_theta*np.cos(9*phi)
    elif phase_factor == 10:
        return norm_factor[10]*654729075.0*sin_theta**10*np.cos(10*phi)
    else:
        raise(ValueError("\n  Invalid spherical harmonics expansion index!!!"))
