# portions of code adapted from https://github.com/ellisdg/frangi3d (MIT license)
from itertools import combinations_with_replacement

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage as ndi

from foa3d.utils import (create_background_mask, create_memory_map,
                         divide_nonzero, hsv_orient_cmap, rgb_orient_cmap)


def analyze_hessian_eigen(img, sigma, trunc=4):
    """
    Return the eigenvalues of the local Hessian matrices
    of the input image array, sorted by absolute value (in ascending order),
    along with the related eigenvectors.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    sigma: int
        spatial scale [px]

    trunc: int
        truncate the Gaussian smoothing kernel at this many standard deviations

    Returns
    -------
    eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)

    dom_eigenvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvectors related to the dominant (minimum) eigenvalue
    """
    # compute scaled Hessian matrices
    hessian = compute_scaled_hessian(img, sigma=sigma, trunc=trunc)

    # compute Hessian eigenvalues and orientation vectors
    # related to the dominant one
    eigenval, dom_eigenvec = compute_dominant_eigen(hessian)

    return eigenval, dom_eigenvec


def compute_dominant_eigen(hessian):
    """
    Compute the eigenvalues (sorted by absolute value)
    of the symmetrical Hessian matrix.

    Parameters
    ----------
    hessian: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        input array of local Hessian matrices

    Returns
    -------
    sorted_eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)

    dom_eigenvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvectors related to the dominant (minimum) eigenvalue
    """
    # compute and sort the eigenvalues/eigenvectors
    # of the image Hessian matrices
    eigenval, eigenvec = np.linalg.eigh(hessian)
    sorted_eigenval, sorted_eigenvec = sort_eigen(eigenval, eigenvec)

    # select the eigenvectors related to dominant eigenvalues
    dom_eigenvec = sorted_eigenvec[..., 0]

    return sorted_eigenval, dom_eigenvec


def compute_fractional_anisotropy(eigenval):
    """
    Compute structure tensor fractional anisotropy
    as in Schilling et al. (2018).

    Parameters
    ----------
    eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        structure tensor eigenvalues (at the best local spatial scale)

    Returns
    -------
    fa: numpy.ndarray (shape=(3,), dtype=float)
        fractional anisotropy
    """
    fa = np.sqrt(0.5 * divide_nonzero(
                 np.square((eigenval[..., 0] - eigenval[..., 1])) +
                 np.square((eigenval[..., 0] - eigenval[..., 2])) +
                 np.square((eigenval[..., 1] - eigenval[..., 2])),
                 np.sum(eigenval ** 2, axis=-1)))

    return fa


def compute_frangi_features(eigen1, eigen2, eigen3, gamma):
    """
    Compute the basic image features employed by the Frangi filter.

    Parameters
    ----------
    eigen1: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    gamma: float
        background score sensitivity

    Returns
    -------
    ra: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        plate-like object score

    rb: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        blob-like object score

    s: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
       second-order structureness

    gamma: float
        background score sensitivity
        (automatically computed if not provided as input)
    """
    ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))
    rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))
    s = compute_structureness(eigen1, eigen2, eigen3)

    # compute default gamma sensitivity
    if gamma is None:
        gamma = 0.5 * np.max(s)

    return ra, rb, s, gamma


def compute_scaled_hessian(img, sigma=1, trunc=4):
    """
    Computes the scaled and normalized Hessian matrices of the input image.
    This is then used to estimate Frangi's vesselness probability score.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    sigma: int
        spatial scale [px]

    trunc: int
        truncate the Gaussian smoothing kernel at this many standard deviations

    Returns
    -------
    hessian: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        Hessian matrix of image second derivatives
    """
    # scale selection
    scaled_img = ndi.gaussian_filter(img, sigma=sigma, output=np.float32, truncate=trunc)

    # compute the first order gradients
    gradient_list = np.gradient(scaled_img)

    # compute the Hessian matrix elements
    hessian_elements = [np.gradient(gradient_list[ax0], axis=ax1)
                        for ax0, ax1 in combinations_with_replacement(range(img.ndim), 2)]

    # scale the elements of the Hessian matrix
    corr_factor = sigma ** 2
    hessian_elements = [corr_factor * element for element in hessian_elements]

    # create the Hessian matrix from its basic elements
    hessian = np.zeros((img.ndim, img.ndim) + scaled_img.shape, dtype=scaled_img.dtype)
    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(img.ndim), 2)):
        element = hessian_elements[index]
        hessian[ax0, ax1, ...] = element
        if ax0 != ax1:
            hessian[ax1, ax0, ...] = element

    # re-arrange axes
    hessian = np.moveaxis(hessian, (0, 1), (-2, -1))

    return hessian


def compute_plate_like_score(ra, alpha):
    return 1 - np.exp(np.divide(np.negative(np.square(ra)), 2 * np.square(alpha)))


def compute_blob_like_score(rb, beta):
    return np.exp(np.divide(np.negative(np.square(rb)), 2 * np.square(beta)))


def compute_background_score(s, gamma):
    return 1 - np.exp(np.divide(np.negative(np.square(s)), 2 * np.square(gamma)))


def compute_structureness(eigen1, eigen2, eigen3):
    return np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))


def compute_scaled_vesselness(eigen1, eigen2, eigen3, alpha, beta, gamma):
    """
    Estimate Frangi's vesselness probability.

    Parameters
    ----------
    eigen1: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    Returns
    -------
    vesselness: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image
    """
    ra, rb, s, gamma = compute_frangi_features(eigen1, eigen2, eigen3, gamma)
    plate = compute_plate_like_score(ra, alpha)
    blob = compute_blob_like_score(rb, beta)
    background = compute_background_score(s, gamma)
    vesselness = plate * blob * background

    return vesselness


def compute_scaled_orientation(scale_px, img, alpha=0.001, beta=1, gamma=None, dark=False):
    """
    Compute fiber orientation vectors at the input spatial scale of interest

    Parameters
    ----------
    scale_px: int
        spatial scale [px]

    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    dark: bool
        if True, enhance black 3D tubular structures
        (i.e., negative contrast polarity)

    Returns
    -------
    frangi_img: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image

    eigenvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        3D orientation map at the input spatial scale

    eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)
    """
    # Hessian matrix estimation and eigenvalue decomposition
    eigenval, eigenvec = analyze_hessian_eigen(img, scale_px)

    # compute Frangi's vesselness probability
    eigen1, eigen2, eigen3 = eigenval
    vesselness = compute_scaled_vesselness(eigen1, eigen2, eigen3, alpha=alpha, beta=beta, gamma=gamma)

    # reject vesselness background
    frangi_img = reject_vesselness_background(vesselness, eigen2, eigen3, dark)

    # stack eigenvalues list
    eigenval = np.stack(eigenval, axis=-1)

    return frangi_img, eigenvec, eigenval


def compute_parall_scaled_orientation(scales_px, img, alpha=0.001, beta=1, gamma=None, dark=False):
    """
    Compute fiber orientation vectors over the spatial scales of interest using concurrent workers.

    Parameters
    ----------
    scales_px: numpy.ndarray (dtype=float)
        Frangi filter scales [px]

    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    dark: bool
        if True, enhance black 3D tubular structures
        (i.e., negative contrast polarity)

    Returns
    -------
    frangi_img: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image

    eigenvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        3D orientation map at the input spatial scale

    eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)
    """
    n_scales = len(scales_px)
    with Parallel(n_jobs=n_scales, prefer='threads', require='sharedmem') as parallel:
        par_res = parallel(
                    delayed(compute_scaled_orientation)(
                        scales_px[i], img=img,
                        alpha=alpha, beta=beta, gamma=gamma, dark=dark) for i in range(n_scales))

        # unpack and stack results
        enh_img_tpl, eigenvec_tpl, eigenval_tpl = zip(*par_res)
        eigenval = np.stack(eigenval_tpl, axis=0)
        eigenvec = np.stack(eigenvec_tpl, axis=0)
        frangi = np.stack(enh_img_tpl, axis=0)

        # get max scale-wise vesselness
        best_idx = np.expand_dims(np.argmax(frangi, axis=0), axis=0)
        frangi = np.take_along_axis(frangi, best_idx, axis=0).squeeze(axis=0)

        # select fiber orientation vectors (and the associated eigenvalues) among different scales
        best_idx = np.expand_dims(best_idx, axis=-1)
        eigenval = np.take_along_axis(eigenval, best_idx, axis=0).squeeze(axis=0)
        fbr_vec = np.take_along_axis(eigenvec, best_idx, axis=0).squeeze(axis=0)

    return frangi, fbr_vec, eigenval


def init_frangi_arrays(cfg, img_shp, slc_shp, rsz_ratio, tmp_dir, msk_bc=False):
    """
    Initialize the output datasets of the Frangi filtering stage.

    Parameters
    ----------
    cfg: dict
        Frangi filter configuration

            alpha: float
                plate-like score sensitivity

            beta: float
                blob-like score sensitivity

            gamma: float
                background score sensitivity

            scales_px: numpy.ndarray (dtype=float)
                Frangi filter scales [px]

            scales_um: numpy.ndarray (dtype=float)
                Frangi filter scales [μm]

            smooth_sd: numpy.ndarray (shape=(3,), dtype=int)
                3D standard deviation of the smoothing Gaussian filter [px]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            z_rng: int
                output z-range in [px]

            bc_ch: int
                neuronal bodies channel

            fb_ch: int
                myelinated fibers channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            hsv_cmap: bool
                generate HSV colormap of 3D fiber orientations

            exp_all: bool
                export all images

    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        3D image shape [px]

    slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    tmp_dir: str
        temporary file directory

    msk_bc: bool
        if True, mask neuronal bodies
        in the optionally provided image channel

    Returns
    -------
    out_img: dict
        output image dictionary

        vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
            initialized fiber orientation 3D image

        clr: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=uint8)
            initialized orientation colormap image

        fa: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized fractional anisotropy image

        frangi: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized Frangi-enhanced image

        iso: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized fiber image (isotropic resolution)

        fbr_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized fiber mask image

        bc_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            initialized soma mask image

    z_sel: NumPy slice object
        selected z-depth range
    """
    # shape copies
    img_shp = img_shp.copy()
    slc_shp = slc_shp.copy()

    # adapt output z-axis shape if required
    z_min, z_max = cfg['z_rng']
    if z_min != 0 or z_max is not None:
        if z_max is None:
            z_max = slc_shp[0]
        img_shp[0] = z_max - z_min
    z_sel = slice(z_min, z_max, 1)

    # output shape
    img_dims = len(img_shp)
    tot_shp = tuple(np.ceil(rsz_ratio * img_shp).astype(int))

    # fiber channel arrays
    iso_fbr_img = create_memory_map(tot_shp, dtype='uint8', name='iso_fiber', tmp_dir=tmp_dir)
    if cfg['exp_all']:
        frangi_img = create_memory_map(tot_shp, dtype='uint8', name='frangi', tmp_dir=tmp_dir)
        fbr_msk = create_memory_map(tot_shp, dtype='uint8', name='fbr_msk', tmp_dir=tmp_dir)
        fa_img = create_memory_map(tot_shp, dtype='float32', name='fa', tmp_dir=tmp_dir)

        # soma channel array
        if msk_bc:
            bc_msk = create_memory_map(tot_shp, dtype='uint8', name='bc_msk', tmp_dir=tmp_dir)
        else:
            bc_msk = None
    else:
        frangi_img, fbr_msk, fa_img, bc_msk = (None, None, None, None)

    # fiber orientation arrays
    vec_shape = tot_shp + (img_dims,)
    fbr_vec_img = create_memory_map(vec_shape, dtype='float32', name='fiber_vec', tmp_dir=tmp_dir)
    fbr_vec_clr = create_memory_map(vec_shape, dtype='uint8', name='fiber_cmap', tmp_dir=tmp_dir)

    # fill output image dictionary
    out_img = dict()
    out_img['vec'] = fbr_vec_img
    out_img['clr'] = fbr_vec_clr
    out_img['fa'] = fa_img
    out_img['frangi'] = frangi_img
    out_img['iso'] = iso_fbr_img
    out_img['fbr_msk'] = fbr_msk
    out_img['bc_msk'] = bc_msk

    return out_img, z_sel


def frangi_filter(img, scales_px=1, alpha=0.001, beta=1.0, gamma=None, dark=False, hsv=False):
    """
    Apply 3D Frangi filter to 3D microscopy image.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    scales_px: int or numpy.ndarray (dtype=int)
        analyzed spatial scales [px]

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity (if None, gamma is automatically tailored)

    dark: bool
        if True, enhance black 3D tubular structures
        (i.e., negative contrast polarity)

    hsv: bool

    Returns
    -------
    orient_slc: dict
        slice orientation dictionary

        vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
            3D fiber orientation field

        clr_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
            orientation colormap image

        fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
            fractional anisotropy image

    frangi: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image
    """
    # single-scale vesselness analysis
    if len(scales_px) == 1:
        frangi, fbr_vec, eigenval = \
            compute_scaled_orientation(scales_px[0], img, alpha=alpha, beta=beta, gamma=gamma, dark=dark)

    # parallel scaled vesselness analysis
    else:
        frangi, fbr_vec, eigenval = \
            compute_parall_scaled_orientation(scales_px, img, alpha=alpha, beta=beta, gamma=gamma, dark=dark)

    # generate fractional anisotropy image
    fa = compute_fractional_anisotropy(eigenval)

    # generate fiber orientation color map
    fbr_clr = hsv_orient_cmap(fbr_vec) if hsv else rgb_orient_cmap(fbr_vec)

    # fill orientation dictionary
    orient_slc = dict()
    orient_slc['vec_slc'] = fbr_vec
    orient_slc['clr_slc'] = fbr_clr
    orient_slc['fa_slc'] = fa

    return orient_slc, frangi


def mask_background(img, orient, method='yen', invert=False, ts_msk=None):
    """
    Mask fiber orientation arrays.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        fiber (or neuron) fluorescence 3D image

    orient: dict
        slice orientation dictionary

            vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
                fiber orientation vector slice

            clr_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap slice

            fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
                fractional anisotropy slice

    method: str
        thresholding method (refer to skimage.filters)

    invert: bool
        mask inversion flag

    ts_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    Returns
    -------
    orient: dict
        (masked) slice orientation dictionary

            vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
                fiber orientation vector slice (masked)

            clr_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap slice (masked)

            fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
                fractional anisotropy slice (masked)

    bg: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
        background mask
    """
    # generate background mask
    bg = create_background_mask(img, method=method)

    # apply tissue reconstruction mask, when provided
    if ts_msk is not None:
        bg = np.logical_or(bg, np.logical_not(ts_msk))

    # invert mask
    if invert:
        bg = np.logical_not(bg)

    # apply mask to input orientation data dictionary
    for key in orient.keys():
        if orient[key].ndim == 3:
            orient[key][bg] = 0
        else:
            orient[key][bg, :] = 0

    return orient, bg


def reject_vesselness_background(vesselness, eigen2, eigen3, dark):
    """
    Reject the fiber background, exploiting the sign of the "secondary"
    eigenvalues λ2 and λ3.

    Parameters
    ----------
    vesselness: numpy.ndarray (axis order=(Z,Y,X))
        Frangi's vesselness likelihood image

    eigen2: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    dark: bool
        if True, enhance black 3D tubular structures
        (i.e., negative contrast polarity)

    Returns
    -------
    vesselness: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        masked Frangi's vesselness likelihood image
    """
    bg_msk = np.logical_or(eigen2 < 0, eigen3 < 0) if dark else np.logical_or(eigen2 > 0, eigen3 > 0)
    bg_msk = np.logical_or(bg_msk, np.isnan(vesselness))
    vesselness[bg_msk] = 0

    return vesselness


def sort_eigen(eigenval, eigenvec, axis=-1):
    """
    Sort eigenvalue and related eigenvector arrays
    by absolute value along the given axis.

    Parameters
    ----------
    eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        original eigenvalue array

    eigenvec: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        original eigenvector array

    axis: int
        sorted axis

    Returns
    -------
    sorted_eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        sorted eigenvalue array (ascending order)

    sorted_eigenvec: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        sorted eigenvector array
    """
    # sort the eigenvalue array by absolute value (ascending order)
    sorted_val_index = np.abs(eigenval).argsort(axis)
    sorted_eigenval = np.take_along_axis(eigenval, sorted_val_index, axis)
    sorted_eigenval = [np.squeeze(eigenval, axis=axis)
                       for eigenval in np.split(sorted_eigenval, sorted_eigenval.shape[axis], axis=axis)]

    # sort eigenvectors consistently
    sorted_vec_index = sorted_val_index[:, :, :, np.newaxis, :]
    sorted_eigenvec = np.take_along_axis(eigenvec, sorted_vec_index, axis)

    return sorted_eigenval, sorted_eigenvec


def write_frangi_arrays(out_rng, z_sel, iso_slc, frangi_slc, fbr_msk_slc, bc_msk_slc, vec_slc, clr_slc, fa_slc,
                        vec, clr, fa, frangi, iso, fbr_msk, bc_msk):
    """
    Fill the memory-mapped output arrays of the Frangi filter stage.

    Parameters
    ----------
    out_rng: tuple
        3D slice output index range

    z_sel: NumPy slice object
        selected z-depth range

    iso_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image slice

    frangi_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        Frangi-enhanced image slice

    fbr_msk_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image slice

    bc_msk_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        soma mask image slice

    vec_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        fiber orientation vector image slice

    clr_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        orientation colormap image slice

    fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        fractional anisotropy image slice

    vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector field

    clr: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    fa: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced image (fiber probability image)

    iso: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fbr_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    bc_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        soma mask image

    Returns
    -------
    None
    """
    # fill memory-mapped output arrays
    vec_rng_out = tuple(np.append(out_rng, slice(0, 3, 1)))
    vec[vec_rng_out] = vec_slc[z_sel, ...]
    clr[vec_rng_out] = clr_slc[z_sel, ...]
    iso[out_rng] = iso_slc[z_sel, ...].astype(np.uint8)

    # optional output images: Frangi filter response
    if frangi is not None:
        frangi[out_rng] = (255 * frangi_slc[z_sel, ...]).astype(np.uint8)

    # optional output images: fractional anisotropy
    if fa is not None:
        fa[out_rng] = fa_slc[z_sel, ...]

    # optional output images: fiber mask
    if fbr_msk is not None:
        fbr_msk[out_rng] = (255 * (1 - fbr_msk_slc[z_sel, ...])).astype(np.uint8)

    # optional output images: neuronal soma mask
    if bc_msk is not None:
        bc_msk[out_rng] = (255 * bc_msk_slc[z_sel, ...]).astype(np.uint8)
