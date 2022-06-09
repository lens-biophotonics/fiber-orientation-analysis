# portions of code adapted from https://github.com/ellisdg/frangi3d (MIT license)

import multiprocessing as mp
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
from numba import njit
from scipy import ndimage as ndi

from fiber_orientation.utils import divide_nonzero


def frangi_filter(image, sigma_px=1, alpha=0.001, beta=1.0, gamma=None, dark_fibers=True):
    """
    Apply 3D Frangi filter to input image.

    Parameters
    ----------
    image: ndarray (shape=(Z,Y,X))
        input 3D image

    sigma_px: ndarray (or int)
        analyzed spatial scales in [px]

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity (if None, gamma is automatically tailored)

    dark_fibers: bool
        if True, enhance black 3D tubular structures

    Returns
    -------
    enhanced_array: ndarray (shape=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood function

    fiber_vectors: ndarray (shape=(Z,Y,X,3), dtype=float)
        3D fiber orientation map
    """
    # check image dimensions
    dims = image.ndim
    if not dims == 3:
        raise(ValueError("  Only 3D images are supported!!!"))

    # single-scale vesselness analysis
    n_scales = len(sigma_px)
    if n_scales == 1:
        fiber_vectors, enhanced_array \
            = compute_scaled_orientation(sigma_px[0], image, alpha=alpha, beta=beta, gamma=gamma,  dark=dark_fibers)

    # parallel scaled vesselness analysis
    else:
        par_compute_orientation \
            = partial(compute_scaled_orientation, image=image, alpha=alpha, beta=beta, gamma=gamma, dark=dark_fibers)

        with mp.Pool(n_scales) as p:
            eigenvectors_list, enhanced_array_list = zip(*p.map(par_compute_orientation, sigma_px))
        eigenvectors = np.stack(eigenvectors_list, axis=0)
        enhanced_array = np.stack(enhanced_array_list, axis=0)

        # get max scale-wise vesselness
        best_idx = np.argmax(enhanced_array, axis=0)
        best_idx = np.expand_dims(best_idx, axis=0)
        enhanced_array = np.take_along_axis(enhanced_array, best_idx, axis=0).squeeze(axis=0)

        # select dominant eigenvalues (and associated eigenvectors)
        best_idx = np.expand_dims(best_idx, axis=-1)
        fiber_vectors = np.take_along_axis(eigenvectors, best_idx, axis=0).squeeze(axis=0)

    return enhanced_array, fiber_vectors


def compute_scaled_orientation(sigma_px, image, alpha=0.001, beta=1, gamma=None, dark=False):
    """
    Estimate fiber orientation vectors at input spatial scale.

    Parameters
    ----------
    sigma_px: int
        analyzed spatial scales in [px]

    image: ndarray (shape=(Z,Y,X))
        input 3D image

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
    eigenvectors: ndarray (shape=(Z,Y,X,3), dtype=float)
        3D orientation map

    enhanced_array: ndarray (shape=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood function
    """
    # Hessian estimation and eigenvalue decomposition
    eigenvalues, eigenvectors = analyze_hessian_eigen(image, sigma_px)

    # compute Frangi's vesselness probability
    enhanced_array = compute_scaled_vesselness(*eigenvalues, alpha=alpha, beta=beta, gamma=gamma, dark=dark)

    return eigenvectors, enhanced_array


def analyze_hessian_eigen(image, sigma, truncate=4):
    """
    Return the eigenvalues of the local Hessian matrices
    of the input image array, sorted by absolute value (in ascending order),
    along with the related eigenvectors.

    Parameters
    ----------
    image: ndarray (shape=(Z,Y,X))
        input 3D image

    sigma: int
        spatial scale [px]

    truncate: int
        truncate the Gaussian smoothing kernel at this many standard deviations
        (default: 4)

    Returns
    -------
    eigenval: ndarray (shape=(Z,Y,X,3), dtype: float)
        dominant eigenvalues array

    eigenvec: ndarray (shape=(Z,Y,X,3), dtype: float)
        eigenvectors array
    """
    # compute scaled Hessian matrices
    hessian = compute_scaled_hessian(image, sigma=sigma, truncate=truncate)

    # compute dominant eigenvalues and related eigenvectors
    eigenval, eigenvec = compute_dominant_eigen(hessian)

    return eigenval, eigenvec


def compute_scaled_hessian(image, sigma=1, truncate=4):
    """
    Computes the scaled and normalized Hessian matrices of the input image.
    This is then used to estimate the Frangi's vesselness probability score.

    The Hessian matrix at each 3D coordinate of the input image volume
    is given by the following spatial second derivatives:
    [ gzz, gzy, gzx ]
    [ gyz, gyy, gyx ]
    [ gxz, gxy, gxx ]

    Parameters
    ----------
    image: ndarray
        input 3D image

    sigma: int
        spatial scale [pixel]

    truncate: int
        truncate the Gaussian smoothing kernel at this many standard deviations
        (default: 4)

    Returns
    -------
    hessian: ndarray (shape=(Z,Y,X,3,3), dtype=float)
        Hessian matrix of image second derivatives
    """
    # get number of dimensions
    ndim = image.ndim
    
    # scale selection
    scaled_image = ndi.gaussian_filter(image, sigma=sigma, output=np.float32, truncate=truncate)                                       

    # compute the first order gradients
    gradient_list = np.gradient(scaled_image)

    # compute the Hessian elements
    hessian_elements = [np.gradient(gradient_list[ax0], axis=ax1)
                        for ax0, ax1 in combinations_with_replacement(range(ndim), 2)]

    # scale the elements of the Hessian matrix
    corr_factor = sigma ** 2
    hessian_elements = [corr_factor * element for element in hessian_elements]

    # create Hessian matrix from Hessian elements
    hessian = np.zeros((ndim, ndim) + scaled_image.shape, dtype=scaled_image.dtype)
    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(ndim), 2)):
        element = hessian_elements[index]
        hessian[ax0, ax1, ...] = element        
        if ax0 != ax1:
            hessian[ax1, ax0, ...] = element

    # re-arrange axes
    hessian = np.moveaxis(hessian, (0, 1), (-2, -1))

    return hessian


def compute_dominant_eigen(hessian):
    """
    Compute the eigenvalues (sorted by absolute value)
    of the symmetrical Hessian matrix.

    Parameters
    ----------
    hessian: ndarray
        input array of local Hessian matrices

    Returns
    -------
    sorted_eigenval: ndarray (shape=(Z,Y,X,3), dtype: float)
        Hessian eigenvalues sorted by absolute value (ascending order)

    dominant_eigenvec: ndarray (shape=(Z,Y,X,3), dtype: float)
        Hessian eigenvectors related to the dominant (minimum) eigenvalue
    """
    # compute and sort the eigenvalues/eigenvectors
    # of the image Hessian matrices
    eigenval, eigenvec = np.linalg.eigh(hessian)
    sorted_eigenval, sorted_eigenvec = sort_eigen(eigenval, eigenvec)

    # select the eigenvectors related to dominant eigenvalues
    dominant_eigenvec = sorted_eigenvec[..., 0]

    return sorted_eigenval, dominant_eigenvec


def sort_eigen(eigenval, eigenvec, axis=-1):
    """
    Sort eigenvalue/eigenvector arrays by absolute value along the given axis.

    Parameters
    ----------
    eigenval: ndarray (shape=(Z,Y,X,3), dtype: float)
        input eigenvalue array

    eigenvec: ndarray (shape=(Z,Y,X,3,3), dtype: float)
        input eigenvector array

    axis: int
        sorted axis

    Returns
    -------
    sorted_eigenval: ndarray (shape=(Z,Y,X,3), dtype: float)
        sorted eigenvalue array

    sorted_eigenvec: ndarray (shape=(Z,Y,X,3,3), dtype: float)
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


def compute_scaled_vesselness(eigen1, eigen2, eigen3, alpha, beta, gamma, dark):
    """
    Estimate Frangi's vesselness probability.

    Parameters
    ----------
    eigen1: ndarray (shape=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: ndarray (shape=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: ndarray (shape=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

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
    vesselness: ndarray (shape=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood function
    """
    ra, rb, s, gamma = compute_frangi_features(eigen1, eigen2, eigen3, gamma)
    plate = compute_plate_like_score(ra, alpha)
    blob = compute_blob_like_score(rb, beta)
    background = compute_background_score(s, gamma)
    vesselness = reject_background(plate * blob * background, eigen2, eigen3, dark)        

    return vesselness


def config_frangi_scales(scales_um, px_size):
    """
    Compute the Frangi filter scales in pixel.

    Parameters
    ----------
    scales_um: list (dtype=float)
        Frangi filter scales [μm]

    px_size: int
        TPFM pixel size [μm]

    Returns
    -------
    sigmas_px: ndarray (dtype=int)
        spatial scales [px]
    """
    scales_um = np.asarray(scales_um)
    sigmas_px = scales_um / px_size
    if np.any(np.asarray(sigmas_px) < 0.0):
        raise ValueError("  Negative sigma values are not valid!!!")

    return sigmas_px


def compute_frangi_features(eigen1, eigen2, eigen3, gamma):
    """
    Compute the basic image features employed by the Frangi filter.

    Parameters
    ----------
    eigen1: ndarray (shape=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: ndarray (shape=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: ndarray (shape=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    gamma: float
        background score sensitivity

    Returns
    -------
    ra: ndarray (shape=(Z,Y,X), dtype=float)
        plate-like object score

    rb: ndarray (shape=(Z,Y,X), dtype=float)
        blob-like object score

    s: ndarray (shape=(Z,Y,X), dtype=float)
       second-order structureness

    gamma: float
        background score sensitivity
    """
    ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))
    rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))
    s = compute_structureness(eigen1, eigen2, eigen3)

    # compute 'auto' gamma sensitivity
    if gamma is None:
        gamma = 0.5*np.max(s)

    return ra, rb, s, gamma


@njit(cache=True)
def compute_plate_like_score(ra, alpha):
    return 1 - np.exp(np.negative(np.square(ra)) / (2 * np.square(alpha)))


@njit(cache=True)
def compute_blob_like_score(rb, beta):
    return np.exp(np.negative(np.square(rb) / (2 * np.square(beta))))


@njit(cache=True)
def compute_background_score(s, gamma):
    return 1 - np.exp(np.divide(np.negative(np.square(s)), (2 * np.square(gamma))))


@njit(cache=True)
def compute_structureness(eigen1, eigen2, eigen3):
    return np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))


def reject_background(image, eigen2, eigen3, negative_contrast):
    """
    Reject the fiber background, exploiting the sign of the "secondary"
    eigenvalues λ2 and λ3.

    Parameters
    ----------
    image: ndarray (shape=(Z,Y,X))
        input 3D image

    eigen2: ndarray (shape=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: ndarray (shape=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    negative_contrast: bool
        True for a negative contrast polarity

    Returns
    -------
    image: ndarray (shape=(Z,Y,X))
        masked 3D image
    """
    if negative_contrast:
        image[eigen2 < 0] = 0
        image[eigen3 < 0] = 0
    else:
        image[eigen2 > 0] = 0
        image[eigen3 > 0] = 0
    image[np.isnan(image)] = 0

    return image
