import numpy as np

from foa3d.utils import elapsed_time


# slice progress counter
slc_cnt = 0


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
    clr_text = f"\033[38;2;{r};{g};{b}m{text} \033[38;2;255;255;255m"

    return clr_text


def print_flsh(string_to_print=""):
    """
    Print string and flush output data buffer.

    Parameters
    ----------
    string_to_print: str
        string to be printed

    Returns
    -------
    None
    """
    print(string_to_print, flush=True)


def print_blur(sigma_um, psf_fwhm):
    """
    Print the standard deviation of the smoothing Gaussian filter.

    Parameters
    ----------
    sigma_um: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [μm]
        (resolution anisotropy correction)

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D FWHM of the PSF [μm]

    Returns
    -------
    None
    """
    psf_sz = np.max(psf_fwhm)
    print_flsh(f"Gaussian blur σ      [μm]: ({sigma_um[0]:.3f}, {sigma_um[1]:.3f}, {sigma_um[2]:.3f})\n" +
               f"Adjusted PSF FWHM    [μm]: ({psf_sz:.3f}, {psf_sz:.3f}, {psf_sz:.3f})")


def print_frangi_info(in_img, frangi_cfg, in_slc_shp_um, tot_slc_num):
    """
    Print Frangi filter stage heading.

    Parameters
    ----------
    in_img: dict
        input image dictionary (extended)

            data: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
                3D microscopy image

            ts_msk: numpy.ndarray (dtype=bool)
                tissue reconstruction binary mask

            ch_ax: int
                RGB image channel axis (either 1, 3, or None for grayscale images)

            fb_ch: int
                neuronal fibers channel

            bc_ch: int
                brain cell soma channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
                3D FWHM of the PSF [μm]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            name: str
                name of the 3D microscopy image

            is_vec: bool
                vector field flag

            shape: numpy.ndarray (shape=(3,), dtype=int)
                total image shape

            shape_um: numpy.ndarray (shape=(3,), dtype=float)
                total image shape [μm]

            item_sz: int
                image item size [B]

    frangi_cfg: dict
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

    in_slc_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    tot_slc_num: int
        total number of analyzed image slices

    Returns
    -------
    None
    """
    # print Frangi filter sensitivity and scales
    alpha = frangi_cfg['alpha']
    beta = frangi_cfg['beta']
    gamma = frangi_cfg['gamma']

    scales_um = np.asarray(frangi_cfg['scales_um'])
    if gamma is None:
        gamma = 'auto'

    print_flsh(color_text(0, 191, 255, "\n3D Frangi Filter\n") + "\nSensitivity\n" +
               f"• plate-like \u03B1: {alpha:.1e}\n• blob-like  \u03B2: {beta:.1e}\n• background \u03B3: {gamma}\n\n" +
               f"Enhanced scales      [μm]: {scales_um}\nEnhanced diameters   [μm]: {4 * scales_um}\n")

    # print iterative analysis information
    print_slicing_info(in_img['shape_um'], in_slc_shp_um, tot_slc_num, in_img['px_sz'], in_img['item_sz'])

    # print neuron masking info
    print_soma_masking(in_img['msk_bc'])


def print_frangi_progress(info, is_valid, verbose=1):
    """
    Print Frangi filter progress.

    Parameters
    ----------
    info: tuple
        info to be printed out

    is_valid: bool
        image slice validity flag

    verbose: int
        verbosity level (print info only every "verbose" slices)

    Returns
    -------
    None
    """
    global slc_cnt
    slc_cnt += 1

    # print only every N=verbose image slices
    start_time, batch_sz, tot_slc = info
    if (slc_cnt % verbose == 0 and is_valid) or slc_cnt == tot_slc:
        prog_prc = 100 * slc_cnt / tot_slc
        _, hrs, mins, secs = elapsed_time(start_time)
        print_flsh(f"[Parallel(n_jobs={batch_sz})]:\t{slc_cnt}/{tot_slc} done\t|\t" +
                   f"elapsed: {hrs} hrs {mins} min {int(secs)} s\t{prog_prc}%")


def print_image_info(in_img):
    """
    Print information on the input microscopy image (shape, voxel size, PSF size).

    Parameters
    ----------
    in_img: dict
        input image dictionary
        ('img_data': image data, 'ts_msk': tissue sample mask, 'ch_ax': channel axis)

    Returns
    -------
    None
    """
    # get pixel and PSF sizes
    px_sz = in_img['px_sz']
    psf_fwhm = in_img['psf_fwhm']

    # get channel axis (RGB image only)
    ch_ax = in_img['ch_ax']

    # get image shape (ignore channel axis)
    img_shp = in_img['data'].shape
    if ch_ax is not None:
        img_shp = np.delete(img_shp, ch_ax)

    print_flsh("\n                              Z      Y      X\nImage shape          [μm]: " +
               f"({img_shp[0] * px_sz[0]:.1f}, {img_shp[1] * px_sz[1]:.1f}, {img_shp[2] * px_sz[2]:.1f})\n" +
               f"Pixel size           [μm]: ({px_sz[0]:.3f}, {px_sz[1]:.3f}, {px_sz[2]:.3f})\n" +
               f"PSF FWHM             [μm]: ({psf_fwhm[0]:.3f}, {psf_fwhm[1]:.3f}, {psf_fwhm[2]:.3f})")


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
    print_flsh(f"Image loaded in: {mins} min {secs:3.1f} s")


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
    print_flsh(
        color_text(0, 191, 255,
                   f"\n3D ODF Analysis\n\nResolution   [μm]: {odf_scales_um}\nExpansion degrees: {odf_degrees}\n"))


def print_output_res(px_sz_iso):
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
    print_flsh(f"Adjusted pixel size  [μm]: ({px_sz_iso[0]:.3f}, {px_sz_iso[1]:.3f}, {px_sz_iso[2]:.3f})\n")


def print_pipeline_heading():
    """
    Print Foa3D tool heading.

    Returns
    -------
    None
    """
    print_flsh(color_text(0, 250, 154, "\n3D Fiber Orientation Analysis"))


def print_prepro_heading():
    """
    Print preprocessing heading.

    Returns
    -------
    None
    """
    print_flsh(color_text(0, 191, 255, "\n\nMicroscopy Image Preprocessing\n") +
               "\n                              Z      Y      X")


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
    print_flsh("\n                              Z      Y      X")
    print_flsh(f"Total image shape    [μm]: ({img_shp_um[0]:.1f}, {img_shp_um[1]:.1f}, {img_shp_um[2]:.1f})\n" +
               f"Total image size     [MB]: {np.ceil(img_sz / 1024**2).astype(int)}\n\n" +
               f"Image slice shape    [μm]: ({slc_shp_um[0]:.1f}, {slc_shp_um[1]:.1f}, {slc_shp_um[2]:.1f})\n" +
               f"Image slice size     [MB]: {np.ceil(max_slc_sz / 1024**2).astype(int)}\n" +
               f"Image slice number:        {tot_slc_num}\n")


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
    if msk_bc:
        print_flsh(f'{prt}active\n')
    else:
        print(f'{prt}not active\n')
