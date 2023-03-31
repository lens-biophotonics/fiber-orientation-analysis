import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from foa3d.printing import color_text, print_prepro_heading
from foa3d.utils import fwhm_to_sigma


def config_anisotropy_correction(px_size, psf_fwhm, vector):
    """
    Scanning and light-sheet fluorescence microscopes provide 3D data
    characterized by a lower resolution along the optical axis
    (i.e. the z-axis). However, the longitudinal anisotropy of the PSF
    introduces a strong bias in the estimated 3D orientations
    as discussed by Morawski et al. (NeuroImage, 2018).
    Thus, for obtaining a uniform 3D resolution, the X and Y axes of the input
    microscopy volume images need in general to be blurred.

    Parameters
    ----------
    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D FWHM of the PSF [μm]

    vector: bool

    Returns
    -------
    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        new isotropic pixel size [μm]
    """
    # print preprocessing heading
    if not vector:
        print_prepro_heading()

    # set the isotropic pixel resolution equal to the z-sampling step
    px_size_iso = px_size[0] * np.ones(shape=(3,))

    # adjust PSF anisotropy via lateral Gaussian blurring
    if not np.all(psf_fwhm == psf_fwhm[0]):

        # estimate the PSF variance from input FWHM values [μm**2]
        psf_var = np.square(fwhm_to_sigma(psf_fwhm))

        # estimate the in-plane filter variance [μm**2]
        gauss_var_x = psf_var[0] - psf_var[2]
        gauss_var_y = psf_var[0] - psf_var[1]

        # ...and the corresponding standard deviation [px]
        gauss_sigma_x = np.sqrt(gauss_var_x) / px_size[2]
        gauss_sigma_y = np.sqrt(gauss_var_y) / px_size[1]
        gauss_sigma_z = 0
        smooth_sigma = np.array([gauss_sigma_z, gauss_sigma_y, gauss_sigma_x])

        # print preprocessing info
        gauss_sigma_um = np.multiply(smooth_sigma, px_size)
        if not vector:
            print(color_text(0, 191, 255, "\n(lateral PSF degradation)"))
            print("\n                              Z      Y      X")
            print("Gaussian blur  \u03C3     [μm]: ({0:.3f}, {1:.3f}, {2:.3f})"
                  .format(gauss_sigma_um[0], gauss_sigma_um[1], gauss_sigma_um[2]), end='\r')

    # (no blurring)
    else:
        print("\n")
        smooth_sigma = None

    # print pixel resize info
    if not vector:
        print("\nOriginal pixel size  [μm]: ({0:.3f}, {1:.3f}, {2:.3f})"
              .format(px_size[0], px_size[1], px_size[2]))
        print("Adjusted pixel size  [μm]: ({0:.3f}, {1:.3f}, {2:.3f})\n"
              .format(px_size_iso[0], px_size_iso[1], px_size_iso[2]))

    return smooth_sigma, px_size_iso


def correct_image_anisotropy(img, resize_ratio,
                             sigma=None, pad_mat=None, pad_mode='reflect', anti_aliasing=True, truncate=4):
    """
    Smooth the input volume image along the X and Y axes so that the lateral
    and longitudinal sizes of the optical system's PSF become equal.
    Downsample data in the XY plane in order to uniform the 3D pixel size.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    resize_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D resize ratio

    sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    pad_mat: numpy.ndarray (shape=(3,2), dtype=int)
        padding range array

    pad_mode: str
        data padding mode adopted for the Gaussian filter

    anti_aliasing: bool
        if True, apply an anti-aliasing filter when downsampling the XY plane

    truncate: int
        truncate the Gaussian kernel at this many standard deviations

    Returns
    -------
    iso_img: numpy.ndarray (shape=(Z,Y,X))
        isotropic microscopy volume image
    """
    # no resizing
    if np.all(resize_ratio == 1):
        iso_img = img
    else:
        # get original volume shape
        img_shape = img.shape

        # lateral blurring
        if sigma is not None:
            img = gaussian_filter(img, sigma=sigma, mode=pad_mode, truncate=truncate, output=np.float32)

        # delete padded boundaries
        if pad_mat is not None:
            if np.count_nonzero(pad_mat) > 0:
                img = img[pad_mat[0, 0]:img_shape[0] - pad_mat[0, 1],
                          pad_mat[1, 0]:img_shape[1] - pad_mat[1, 1],
                          pad_mat[2, 0]:img_shape[2] - pad_mat[2, 1]]

        # lateral downsampling
        iso_shape = np.ceil(np.multiply(np.asarray(img.shape), resize_ratio)).astype(int)
        iso_img = np.zeros(shape=iso_shape, dtype=img.dtype)
        for z in range(iso_shape[0]):
            iso_img[z, ...] = resize(img[z, ...], output_shape=tuple(iso_shape[1:]),
                                     anti_aliasing=anti_aliasing, preserve_range=True)

    return iso_img
