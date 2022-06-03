import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from modules.printing import colored
from modules.utils import fwhm_to_sigma


def config_anisotropy_correction(px_size, psf_fwhm):
    """
    Confocal laser scanning microscopes and light-sheet microscopes provide
    3D data with a lower resolution along the optical axis direction
    (i.e. the z-axis).
    However, the longitudinal PSF anisotropy introduces a strong bias
    in the estimated 3D orientations [Morawski et al. 2018].

    Thus, for obtaining a uniform resolution over the 3 dimensions of the input
    TPFM image volumes, the X and Y axes need in general to be blurred.
    The in-plane standard deviation of the applied low-pass Gaussian kernel
    will depend on the PSF size (i.e., on the input FWHM values) as follows:

    PSF_sigma_x**2 + gauss_sigma_x**2 = PSF_sigma_z**2
    PSF_sigma_y**2 + gauss_sigma_y**2 = PSF_sigma_z**2

    Parameters
    ----------
    px_size: ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: ndarray (shape=(3,), dtype=float)
        3D PSF FWHM in [μm]

    Returns
    -------
    smooth_sigma: ndarray (shape=(3,), dtype=int)
        3D standard deviation of low-pass Gaussian filter [px]

    px_size_iso: ndarray (shape=(3,), dtype=float)
        new isotropic spatial sampling [μm]
    """
    # get preprocessing stage configuration (resolution anisotropy correction)
    print(colored(0, 191, 255, "\n\n  TPFM Volume Preprocessing"), end='\r')

    # set the isotropic pixel resolution equal to the z sampling step
    px_size_iso = px_size[0]*np.ones(shape=(3,))

    # adjust PSF anisotropy via
    # transverse Gaussian blurring...
    if not np.all(psf_fwhm == psf_fwhm[0]):

        # estimate the PSF variance from input FWHM values [μm**2]
        psf_var = np.square(fwhm_to_sigma(psf_fwhm))

        # estimate the in-plane filter variance [μm**2]
        gauss_var_x = psf_var[0] - psf_var[2]
        gauss_var_y = psf_var[0] - psf_var[1]

        # ...and standard deviation [pixel]
        gauss_sigma_x = np.sqrt(gauss_var_x) / px_size[2]
        gauss_sigma_y = np.sqrt(gauss_var_y) / px_size[1]
        gauss_sigma_z = 0
        smooth_sigma = np.array([gauss_sigma_z, gauss_sigma_y, gauss_sigma_x])

        # print preprocessing info
        gauss_sigma_um = np.multiply(smooth_sigma, px_size)
        print(colored(0, 191, 255, "\n  (lateral PSF degradation)"))
        print("\n                                Z      Y      X")
        print("  Gaussian blur  \u03C3     [μm]: ({0:.3f}, {1:.3f}, {2:.3f})"
              .format(gauss_sigma_um[0], gauss_sigma_um[1], gauss_sigma_um[2]),
              end='\r')

    # (no blurring)
    else:
        print("\n")
        smooth_sigma = None

    # print pixel resize info
    print("\n  Original pixel size  [μm]: ({0:.3f}, {1:.3f}, {2:.3f})"
          .format(px_size[0], px_size[1], px_size[2]))
    print("  Adjusted pixel size  [μm]: ({0:.3f}, {1:.3f}, {2:.3f})\n"
          .format(px_size_iso[0], px_size_iso[1], px_size_iso[2]))

    return smooth_sigma, px_size_iso


def correct_tpfm_anisotropy(volume, resize_ratio, sigma=None, pad_mat=None,
                            pad_mode='reflect', anti_aliasing=True,
                            truncate=4):
    """
    Smooth the input image volume along the lateral XY axes so that the lateral
    size of the PSF becomes equal to the PSF's depth.
    Downsample data in the XY plane in order to uniform the 3D pixel size.

    Parameters
    ----------
    volume: ndarray (shape=(Z,Y,X))
        input TPFM image volume

    resize_ratio: ndarray (shape=(3,), dtype=float)
        3D axes resize ratio

    sigma: ndarray (shape=(3,), dtype=int)
        3D standard deviation of the blurring Gaussian filter [px]

    pad_mat: ndarray
        padding range array

    pad_mode: string
        data padding mode adopted for the Gaussian filter

    anti_aliasing: bool
        if True, apply anti-aliasing filter when downsampling the XY plane

    truncate: int
        truncate the Gaussian kernel at this many standard deviations
        (default: 4)

    Returns
    -------
    iso_volume: ndarray (shape=(Z,Y,X))
        isotropic TPFM image volume
    """
    # no resizing
    if np.all(resize_ratio == 1):
        iso_volume = volume
    else:
        # get original volume shape
        volume_shape = volume.shape

        # TPFM volume lateral blurring
        if sigma is not None:
            volume = gaussian_filter(volume, sigma=sigma, mode=pad_mode,
                                     truncate=truncate, output=np.float32)

        # delete padded boundaries
        if pad_mat is not None:
            if np.count_nonzero(pad_mat) > 0:
                volume = volume[pad_mat[0, 0]:volume_shape[0]-pad_mat[0, 1],
                                pad_mat[1, 0]:volume_shape[1]-pad_mat[1, 1],
                                pad_mat[2, 0]:volume_shape[2]-pad_mat[2, 1]]

        # TPFM volume lateral downsampling
        iso_shape = np.ceil(np.multiply(np.asarray(volume.shape),
                                        resize_ratio)).astype(int)
        iso_volume = np.zeros(shape=iso_shape, dtype=volume.dtype)
        for z in range(iso_shape[0]):
            iso_volume[z, ...] = resize(volume[z, ...],
                                        output_shape=tuple(iso_shape[1:]),
                                        anti_aliasing=anti_aliasing,
                                        preserve_range=True)

    return iso_volume
