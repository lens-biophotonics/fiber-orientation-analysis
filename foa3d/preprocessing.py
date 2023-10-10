import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from foa3d.printing import print_blur, print_new_res, print_prepro_heading
from foa3d.utils import fwhm_to_sigma


def config_anisotropy_correction(px_sz, psf_fwhm):
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
    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D FWHM of the PSF [μm]

    Returns
    -------
    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    px_sz_iso: numpy.ndarray (shape=(3,), dtype=float)
        new isotropic pixel size [μm]
    """

    # print preprocessing heading
    print_prepro_heading()

    # set the isotropic pixel resolution equal to the z-sampling step
    px_sz_iso = px_sz[0] * np.ones(shape=(3,))

    # adjust PSF anisotropy via lateral Gaussian blurring
    if not np.all(psf_fwhm == psf_fwhm[0]):

        # estimate the PSF variance from input FWHM values [μm^2]
        psf_var = np.square(fwhm_to_sigma(psf_fwhm))

        # estimate the in-plane filter variance [μm^2]
        gauss_var = np.array([0, psf_var[0] - psf_var[1], psf_var[0] - psf_var[2]])

        # ...and the corresponding standard deviation [px]
        smooth_sigma = np.divide(np.sqrt(gauss_var), px_sz)

        # print preprocessing info
        smooth_sigma_um = np.multiply(smooth_sigma, px_sz)
        print_blur(smooth_sigma_um)

    # (no blurring)
    else:
        print("\n")
        smooth_sigma = None

    # print pixel resize info
    print_new_res(px_sz_iso, psf_fwhm)

    return smooth_sigma, px_sz_iso


def correct_image_anisotropy(img, rsz_ratio, sigma=None, pad=None, smooth_pad_mode='reflect', anti_alias=True, trunc=4):
    """
    Smooth the input volume image along the X and Y axes so that the lateral
    and longitudinal sizes of the optical system's PSF become equal.
    Downsample data in the XY plane in order to uniform the 3D pixel size.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        microscopy volume image

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D resize ratio

    sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    pad: numpy.ndarray (shape=(3,2), dtype=int)
        image padding array to be resized

    smooth_pad_mode: str
        image padding mode adopted for the smoothing Gaussian filter

    anti_alias: bool
        if True, apply an anti-aliasing filter when downsampling the XY plane

    trunc: int
        truncate the Gaussian kernel at this many standard deviations

    Returns
    -------
    iso_img: numpy.ndarray (axis order=(Z,Y,X))
        isotropic microscopy volume image

    rsz_pad: numpy.ndarray (shape=(3,2), dtype=int)
        resized image padding array
    """
    # no resizing
    if np.all(rsz_ratio == 1):
        return img, pad

    # lateral blurring
    else:
        if sigma is not None:
            img = gaussian_filter(img, sigma=sigma, mode=smooth_pad_mode, truncate=trunc, output=np.float32)

        # downsampling
        iso_shape = np.ceil(np.multiply(np.asarray(img.shape), rsz_ratio)).astype(int)
        iso_img = np.zeros(shape=iso_shape, dtype=img.dtype)
        for z in range(iso_shape[0]):
            iso_img[z, ...] = \
                resize(img[z, ...], output_shape=tuple(iso_shape[1:]), anti_aliasing=anti_alias, preserve_range=True)

        # resize padding array accordingly
        if pad is not None:
            rsz_pad = np.floor(np.multiply(np.array([rsz_ratio, rsz_ratio]).transpose(), pad)).astype(int)
            return iso_img, rsz_pad

        else:
            return iso_img, None
