# portions of code adapted from https://github.com/ellisdg/frangi3d (MIT license)
from itertools import combinations_with_replacement

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage as ndi

from foa3d.utils import divide_nonzero


def analyze_hessian_eigen(img, sigma, truncate=4):
    """
    Return the eigenvalues of the local Hessian matrices
    of the input image array, sorted by absolute value (in ascending order),
    along with the related eigenvectors.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    sigma: int
        spatial scale [px]

    truncate: int
        truncate the Gaussian smoothing kernel at this many standard deviations

    Returns
    -------
    eigenval: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)

    dom_eigenvec: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        Hessian eigenvectors related to the dominant (minimum) eigenvalue
    """
    # compute scaled Hessian matrices
    hessian = compute_scaled_hessian(img, sigma=sigma, truncate=truncate)

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
    hessian: numpy.ndarray (shape=(Z,Y,X,3,3), dtype=float)
        input array of local Hessian matrices

    Returns
    -------
    sorted_eigenval: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)

    dom_eigenvec: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        Hessian eigenvectors related to the dominant (minimum) eigenvalue
    """
    # compute and sort the eigenvalues/eigenvectors
    # of the image Hessian matrices
    eigenval, eigenvec = np.linalg.eigh(hessian)
    sorted_eigenval, sorted_eigenvec = sort_eigen(eigenval, eigenvec)

    # select the eigenvectors related to dominant eigenvalues
    dom_eigenvec = sorted_eigenvec[..., 0]

    return sorted_eigenval, dom_eigenvec


def compute_frangi_features(eigen1, eigen2, eigen3, gamma):
    """
    Compute the basic image features employed by the Frangi filter.

    Parameters
    ----------
    eigen1: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    gamma: float
        background score sensitivity

    Returns
    -------
    ra: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        plate-like object score

    rb: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        blob-like object score

    s: numpy.ndarray (shape=(Z,Y,X), dtype=float)
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


def compute_scaled_hessian(img, sigma=1, truncate=4):
    """
    Computes the scaled and normalized Hessian matrices of the input image.
    This is then used to estimate Frangi's vesselness probability score.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    sigma: int
        spatial scale [px]

    truncate: int
        truncate the Gaussian smoothing kernel at this many standard deviations

    Returns
    -------
    hessian: numpy.ndarray (shape=(Z,Y,X,3,3), dtype=float)
        Hessian matrix of image second derivatives
    """
    # get number of dimensions
    ndim = img.ndim

    # scale selection
    scaled_img = ndi.gaussian_filter(img, sigma=sigma, output=np.float32, truncate=truncate)

    # compute the first order gradients
    gradient_list = np.gradient(scaled_img)

    # compute the Hessian matrix elements
    hessian_elements = [np.gradient(gradient_list[ax0], axis=ax1)
                        for ax0, ax1 in combinations_with_replacement(range(ndim), 2)]

    # scale the elements of the Hessian matrix
    corr_factor = sigma ** 2
    hessian_elements = [corr_factor * element for element in hessian_elements]

    # create the Hessian matrix from its basic elements
    hessian = np.zeros((ndim, ndim) + scaled_img.shape, dtype=scaled_img.dtype)
    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(ndim), 2)):
        element = hessian_elements[index]
        hessian[ax0, ax1, ...] = element
        if ax0 != ax1:
            hessian[ax1, ax0, ...] = element

    # re-arrange axes
    hessian = np.moveaxis(hessian, (0, 1), (-2, -1))

    return hessian


def compute_scaled_orientation(scale_px, img, alpha=0.001, beta=1, gamma=None, dark=False):
    """
    Estimate fiber orientation vectors at the input spatial scale.

    Parameters
    ----------
    scale_px: int
        spatial scale [px]

    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

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
    enhanced_img: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image

    eigenvec: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        3D orientation map at the input spatial scale

    eigenval: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)
    """
    # Hessian matrix estimation and eigenvalue decomposition
    eigenval, eigenvec = analyze_hessian_eigen(img, scale_px)

    # compute Frangi's vesselness probability
    eigen1, eigen2, eigen3 = eigenval
    vesselness = compute_scaled_vesselness(eigen1, eigen2, eigen3, alpha=alpha, beta=beta, gamma=gamma)

    # reject vesselness background
    enhanced_img = reject_vesselness_background(vesselness, eigen2, eigen3, dark)

    # stack eigenvalues list
    eigenval = np.stack(eigenval, axis=-1)

    return enhanced_img, eigenvec, eigenval


def compute_scaled_vesselness(eigen1, eigen2, eigen3, alpha, beta, gamma):
    """
    Estimate Frangi's vesselness probability.

    Parameters
    ----------
    eigen1: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    Returns
    -------
    vesselness: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image
    """
    ra, rb, s, gamma = compute_frangi_features(eigen1, eigen2, eigen3, gamma)
    plate = compute_plate_like_score(ra, alpha)
    blob = compute_blob_like_score(rb, beta)
    background = compute_background_score(s, gamma)
    vesselness = plate * blob * background

    return vesselness


def convert_frangi_scales(scales_um, px_size):
    """
    Compute the Frangi filter scales in pixel.

    Parameters
    ----------
    scales_um: list (dtype=float)
        Frangi filter scales [μm]

    px_size: int
        isotropic pixel size [μm]

    Returns
    -------
    scales_px: numpy.ndarray (dtype=int)
        Frangi filter scales [px]
    """
    scales_um = np.asarray(scales_um)
    scales_px = scales_um / px_size

    return scales_px


def frangi_filter(img, scales_px=1, alpha=0.001, beta=1.0, gamma=None, dark=True):
    """
    Apply 3D Frangi filter to microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

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

    Returns
    -------
    enhanced_img: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image

    fiber_vec: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        3D fiber orientation map

    eigenval: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)
    """
    # single-scale vesselness analysis
    n_scales = len(scales_px)
    if n_scales == 1:
        enhanced_img, fiber_vec, eigenval \
            = compute_scaled_orientation(scales_px[0], img, alpha=alpha, beta=beta, gamma=gamma,  dark=dark)

    # parallel scaled vesselness analysis
    else:
        with Parallel(n_jobs=n_scales, backend='threading', max_nbytes=None) as parallel:
            par_res = \
                parallel(
                    delayed(compute_scaled_orientation)(
                        scales_px[i], img=img,
                        alpha=alpha, beta=beta, gamma=gamma, dark=dark) for i in range(n_scales))

            # unpack and stack results
            enhanced_img_tpl, eigenvectors_tpl, eigenvalues_tpl = zip(*par_res)
            eigenval = np.stack(eigenvalues_tpl, axis=0)
            eigenvec = np.stack(eigenvectors_tpl, axis=0)
            enhanced_img = np.stack(enhanced_img_tpl, axis=0)

            # get max scale-wise vesselness
            best_idx = np.argmax(enhanced_img, axis=0)
            best_idx = np.expand_dims(best_idx, axis=0)
            enhanced_img = np.take_along_axis(enhanced_img, best_idx, axis=0).squeeze(axis=0)

            # select fiber orientation vectors (and the associated eigenvalues) among different scales
            best_idx = np.expand_dims(best_idx, axis=-1)
            eigenval = np.take_along_axis(eigenval, best_idx, axis=0).squeeze(axis=0)
            fiber_vec = np.take_along_axis(eigenvec, best_idx, axis=0).squeeze(axis=0)

    return enhanced_img, fiber_vec, eigenval


def reject_vesselness_background(vesselness, eigen2, eigen3, dark):
    """
    Reject the fiber background, exploiting the sign of the "secondary"
    eigenvalues λ2 and λ3.

    Parameters
    ----------
    vesselness: numpy.ndarray (shape=(Z,Y,X))
        Frangi's vesselness likelihood image

    eigen2: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    dark: bool
        if True, enhance black 3D tubular structures
        (i.e., negative contrast polarity)

    Returns
    -------
    vesselness: numpy.ndarray (shape=(Z,Y,X))
        masked Frangi's vesselness likelihood image
    """
    if dark:
        vesselness[eigen2 < 0] = 0
        vesselness[eigen3 < 0] = 0
    else:
        vesselness[eigen2 > 0] = 0
        vesselness[eigen3 > 0] = 0
    vesselness[np.isnan(vesselness)] = 0

    return vesselness


def sort_eigen(eigenval, eigenvec, axis=-1):
    """
    Sort eigenvalue and related eigenvector arrays
    by absolute value along the given axis.

    Parameters
    ----------
    eigenval: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        original eigenvalue array

    eigenvec: numpy.ndarray (shape=(Z,Y,X,3,3), dtype=float)
        original eigenvector array

    axis: int
        sorted axis

    Returns
    -------
    sorted_eigenval: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        sorted eigenvalue array (ascending order)

    sorted_eigenvec: numpy.ndarray (shape=(Z,Y,X,3,3), dtype=float)
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


def compute_plate_like_score(ra, alpha):
    return 1 - np.exp(np.divide(np.negative(np.square(ra)), 2 * np.square(alpha)))


def compute_blob_like_score(rb, beta):
    return np.exp(np.divide(np.negative(np.square(rb)), 2 * np.square(beta)))


def compute_background_score(s, gamma):
    return 1 - np.exp(np.divide(np.negative(np.square(s)), 2 * np.square(gamma)))


def compute_structureness(eigen1, eigen2, eigen3):
    return np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))
