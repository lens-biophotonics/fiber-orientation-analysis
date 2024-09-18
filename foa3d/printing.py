import os
from platform import system

import numpy as np

from foa3d.utils import elapsed_time

# adjust ANSI escape sequence
# decoding to Windows OS
if system == 'Windows':
    os.system("color")


def color_text(r, g, b, text):
    """
    Get colored text string.

    Parameters
    ----------
    r: int
        red channel value

    g: int
        green channel value

    b: int
        blue channel value

    text: str
        text string

    Returns
    -------
    clr_text: str
        colored text
    """
    clr_text = "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

    return clr_text


def print_frangi_info(alpha, beta, gamma, scales_um, img_shp_um, in_slc_shp_um,
                      tot_slc_num, px_sz, img_item_sz, msk_bc):
    """
    Print Frangi filter stage heading.

    Parameters
    ----------
    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    scales_um: list (dtype=float)
        analyzed spatial scales [μm]

    img_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    in_slc_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    tot_slc_num: int
        total number of analyzed image slices

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    img_item_sz: int
        image item size (in bytes)

    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel

    Returns
    -------
    None
    """

    scales_um = np.asarray(scales_um)
    if gamma is None:
        gamma = 'auto'

    print(color_text(0, 191, 255, "\n3D Frangi Filter"))
    print(u"\n\u03B1: {0:.3f}\n".format(alpha)
          + u"\u03B2: {0:.3f}\n".format(beta)
          + u"\u03B3: {0}\n".format(gamma))
    print("Enhanced scales      [\u03BCm]: {}".format(scales_um))
    print("Enhanced diameters   [\u03BCm]: {}\n".format(4 * scales_um))

    # print iterative analysis information
    print_slicing_info(img_shp_um, in_slc_shp_um, tot_slc_num, px_sz, img_item_sz)

    # print neuron masking info
    print_soma_masking(msk_bc)


def print_analysis_time(start_time):
    """
    Print image processing time.

    Parameters
    ----------
    start_time: float
        analysis start time

    Returns
    -------
    None
    """
    _, hrs, mins, secs = elapsed_time(start_time)
    print("\nMicroscopy image analyzed in: {0} min {1:3.1f} s\n".format(mins, secs)) if hrs == 0 else \
        print("\nMicroscopy image analyzed in: {0} hrs {1} min {2:3.1f} s\n".format(hrs, mins, secs))


def print_blur(smooth_sigma_um, psf_fwhm):
    """
    Print the standard deviation of the smoothing Gaussian filter.

    Parameters
    ----------
    smooth_sigma_um: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [μm]
        (resolution anisotropy correction)

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D FWHM of the PSF [μm]

    Returns
    -------
    None
    """
    print("Gaussian blur \u03C3      [μm]: ({0:.3f}, {1:.3f}, {2:.3f})"
          .format(smooth_sigma_um[0], smooth_sigma_um[1], smooth_sigma_um[2]))
    print("Adjusted PSF FWHM    [μm]: ({0:.3f}, {0:.3f}, {0:.3f})".format(np.max(psf_fwhm)))


def print_import_time(start_time):
    """
    Print image import time.

    Parameters
    ----------
    start_time: float
        import start time

    Returns
    -------
    None
    """
    _, _, mins, secs = elapsed_time(start_time)
    print("Image loaded in: {0} min {1:3.1f} s".format(mins, secs))


def print_odf_info(odf_scales_um, odf_degrees):
    """
    Print ODF analysis heading.

    Parameters
    ----------
    odf_scales_um: list (dtype=float)
        fiber ODF resolutions (super-voxel sides [μm])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    Returns
    -------
    None
    """
    print(color_text(0, 191, 255, "\n3D ODF Analysis"))
    print("\nResolution   [\u03BCm]: {}".format(odf_scales_um))
    print("Expansion degrees: {}\n".format(odf_degrees))


def print_pipeline_heading():
    """
    Print Foa3D tool heading.

    Returns
    -------
    None
    """
    print(color_text(0, 250, 154, "\n3D Fiber Orientation Analysis"))


def print_prepro_heading():
    """
    Print preprocessing heading.

    Returns
    -------
    None
    """
    print(color_text(0, 191, 255, "\n\nMicroscopy Image Preprocessing"))
    print("\n                              Z      Y      X")


def print_native_res(px_sz, psf_fwhm):
    """
    Print the native pixel size and optical resolution of the microscopy system.

    Parameters
    ----------
    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        PSF 3D FWHM [μm]

    Returns
    -------
    None
    """
    print("Pixel size           [μm]: ({0:.3f}, {1:.3f}, {2:.3f})".format(px_sz[0], px_sz[1], px_sz[2]))
    print("PSF FWHM             [μm]: ({0:.3f}, {1:.3f}, {2:.3f})".format(psf_fwhm[0], psf_fwhm[1], psf_fwhm[2]))


def print_new_res(px_sz_iso):
    """
    Print the adjusted isotropic spatial resolution.

    Parameters
    ----------
    px_sz_iso: numpy.ndarray (shape=(3,), dtype=float)
        new isotropic pixel size [μm]

    Returns
    -------
    None
    """
    print("Adjusted pixel size  [μm]: ({0:.3f}, {1:.3f}, {2:.3f})\n".format(px_sz_iso[0], px_sz_iso[1], px_sz_iso[2]))


def print_slicing_info(img_shp_um, slc_shp_um, tot_slc_num, px_sz, img_item_sz):
    """
    Print information on the slicing of the basic image sub-volumes processed by the Foa3D tool.

    Parameters
    ----------
    img_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        3D microscopy image [μm]

    slc_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    tot_slc_num: int
        total number of analyzed image slices

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    img_item_sz: int
        image item size (in bytes)

    Returns
    -------
    None
    """

    # adjust slice shape
    if np.any(img_shp_um < slc_shp_um):
        slc_shp_um = img_shp_um

    # get image memory size
    img_sz = img_item_sz * np.prod(np.divide(img_shp_um, px_sz))

    # get slice memory size
    max_slc_sz = img_item_sz * np.prod(np.divide(slc_shp_um, px_sz))

    # print info
    print("\n                              Z      Y      X")
    print("Total image shape    [μm]: ({0:.1f}, {1:.1f}, {2:.1f})".format(img_shp_um[0], img_shp_um[1], img_shp_um[2]))
    print("Total image size     [MB]: {0}\n".format(np.ceil(img_sz / 1024**2).astype(int)))
    print("Image slice shape    [μm]: ({0:.1f}, {1:.1f}, {2:.1f})".format(slc_shp_um[0], slc_shp_um[1], slc_shp_um[2]))
    print("Image slice size     [MB]: {0}".format(np.ceil(max_slc_sz / 1024**2).astype(int)))
    print("Image slice number:        {0}\n".format(tot_slc_num))


def print_soma_masking(msk_bc):
    """
    Print information on the optional masking of neuronal bodies.

    Parameters
    ----------
    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel

    Returns
    -------
    None
    """
    prt = 'Soma mask: '
    print('{}active\n'.format(prt)) if msk_bc else print('{}not active\n'.format(prt))


def print_image_shape(cli_args, img, is_tiled, ch_ax=None):
    """
    Print 3D microscopy image shape.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    is_tiled: bool
        True for tiled reconstructions aligned using ZetaStitcher

    ch_ax: int
        channel axis (if ndim == 4)

    Returns
    -------
    None
    """

    # get pixel size
    px_sz_z = cli_args.px_size_z
    px_sz_xy = cli_args.px_size_xy

    # adapt axis order
    img_shp = img.shape
    if len(img_shp) == 4:
        ch_ax = 1 if is_tiled else -1

    # get image shape
    if ch_ax is not None:
        img_shp = np.delete(img_shp, ch_ax)

    print("\n                              Z      Y      X")
    print("Image shape          [μm]: ({0:.1f}, {1:.1f}, {2:.1f})"
          .format(img_shp[0] * px_sz_z, img_shp[1] * px_sz_xy, img_shp[2] * px_sz_xy))
