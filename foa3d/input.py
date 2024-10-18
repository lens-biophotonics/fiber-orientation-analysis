import argparse

from time import perf_counter
from os import path

import numpy as np
import tifffile as tiff

try:
    from zetastitcher import VirtualFusedVolume
except ImportError:
    pass

from foa3d.output import create_save_dirs
from foa3d.preprocessing import config_anisotropy_correction
from foa3d.printing import (color_text, print_flsh, print_image_info,
                            print_import_time)
from foa3d.utils import (create_background_mask, create_memory_map,
                         detect_ch_axis, get_item_bytes, get_config_label)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def get_cli_parser():
    """
    Parse command line arguments.

    Returns
    -------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments
    """
    # configure parser object
    cli_parser = argparse.ArgumentParser(
        description='Foa3D: A 3D Fiber Orientation Analysis Pipeline\n'
                    'author:     Michele Sorelli (2022)\n'
                    'references: Frangi  et al.  (1998) '
                    'Multiscale vessel enhancement filtering.'
                    ' In Medical Image Computing and'
                    ' Computer-Assisted Intervention 1998, pp. 130-137.\n'
                    '            Alimi   et al.  (2020) '
                    'Analytical and fast Fiber Orientation Distribution '
                    'reconstruction in 3D-Polarized Light Imaging. '
                    'Medical Image Analysis, 65, pp. 101760.\n'
                    '            Sorelli et al.  (2023) '
                    'Fiber enhancement and 3D orientation analysis '
                    'in label-free two-photon fluorescence microscopy. '
                    'Scientific Reports, 13, pp. 4160.\n',
        formatter_class=CustomFormatter)
    cli_parser.add_argument(dest='image_path',
                            help='path to input 3D microscopy image or 4D array of fiber orientation vectors\n'
                                 '* supported formats:\n'
                                 '  - .tif .tiff (microscopy image or fiber orientation vectors)\n'
                                 '  - .yml (ZetaStitcher\'s stitch file of tiled microscopy reconstruction)\n'
                                 '  - .npy (fiber orientation vectors)\n'
                                 '* image axes order:\n'
                                 '  - grayscale image:      (Z, Y, X)\n'
                                 '  - RGB image:            (Z, Y, X, C) or (Z, C, Y, X)\n'
                                 '  - NumPy vector image:   (Z, Y, X, C) or (Z, C, Y, X)\n'
                                 '  - TIFF  vector image:   (Z, Y, X, C) or (Z, C, Y, X)')
    cli_parser.add_argument('-a', '--alpha', type=float, default=0.001,
                            help='Frangi\'s plate-like object sensitivity')
    cli_parser.add_argument('-b', '--beta', type=float, default=1.0,
                            help='Frangi\'s blob-like object sensitivity')
    cli_parser.add_argument('-g', '--gamma', type=float, default=None,
                            help='Frangi\'s background score sensitivity')
    cli_parser.add_argument('-s', '--scales', nargs='+', type=float, default=[1.25],
                            help='list of Frangi filter scales [μm]')
    cli_parser.add_argument('-j', '--jobs', type=int, default=None,
                            help='number of parallel threads used by the Frangi filter stage: '
                                 'use one thread per logical core if None')
    cli_parser.add_argument('-r', '--ram', type=float, default=None,
                            help='maximum RAM available to the Frangi filter stage [GB]: use all if None')
    cli_parser.add_argument('--px-size-xy', type=float, default=0.878, help='lateral pixel size [μm]')
    cli_parser.add_argument('--px-size-z', type=float, default=1.0, help='longitudinal pixel size [μm]')
    cli_parser.add_argument('--psf-fwhm-x', type=float, default=1.0, help='PSF FWHM along horizontal x-axis [μm]')
    cli_parser.add_argument('--psf-fwhm-y', type=float, default=1.0, help='PSF FWHM along vertical y-axis [μm]')
    cli_parser.add_argument('--psf-fwhm-z', type=float, default=1.0, help='PSF FWHM along depth z-axis [μm]')
    cli_parser.add_argument('--fb-ch', type=int, default=1, help='neuronal fibers channel')
    cli_parser.add_argument('--bc-ch', type=int, default=0, help='brain cell soma channel')
    cli_parser.add_argument('--fb-thr', default='li', type=str,
                            help='Frangi filter probability response threshold (t ∈ [0, 1] or skimage.filters method)')
    cli_parser.add_argument('--z-min', type=float, default=0, help='forced minimum output z-depth [μm]')
    cli_parser.add_argument('--z-max', type=float, default=None, help='forced maximum output z-depth [μm]')
    cli_parser.add_argument('--hsv', action='store_true', default=False,
                            help='generate HSV colormap of 3D fiber orientations')
    cli_parser.add_argument('--odf-res', nargs='+', type=float, help='side of the fiber ODF super-voxels: '
                                                                     'do not generate ODFs if None [μm]')
    cli_parser.add_argument('--odf-deg', type=int, default=6,
                            help='degrees of the spherical harmonics series expansion (even number between 2 and 10)')
    cli_parser.add_argument('-o', '--out', type=str, default=None,
                            help='output directory')
    cli_parser.add_argument('-c', '--cell-msk', action='store_true', default=False,
                            help='apply neuronal body mask (the optional channel of neuronal bodies must be available)')
    cli_parser.add_argument('-t', '--tissue-msk', action='store_true', default=False,
                            help='apply tissue background mask')
    cli_parser.add_argument('-v', '--vec', action='store_true', default=False,
                            help='fiber orientation vector image')
    cli_parser.add_argument('-e', '--exp-all', action='store_true', default=False,
                            help='save the full range of images produced by the Frangi filter and ODF stages, '
                                 'e.g. for testing purposes (see documentation)')

    # parse arguments
    cli_args = cli_parser.parse_args()

    return cli_args


def get_image_size(in_img):
    """
    Get information on the input 3D microscopy image.

    Parameters
    ----------
    in_img: dict
        input image dictionary

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

    Returns
    -------
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

    """
    # adapt channel axis
    in_img['shape'] = np.asarray(in_img['data'].shape)
    if in_img['ch_ax'] is None:
        in_img['fb_ch'] = None
        in_img['msk_bc'] = False
    else:
        in_img['shape'] = np.delete(in_img['shape'], in_img['ch_ax'])

    in_img['shape_um'] = np.multiply(in_img['shape'], in_img['px_sz'])
    in_img['item_sz'] = get_item_bytes(in_img['data'])

    return in_img


def get_image_info(cli_args):
    """
    Get microscopy image file path and format.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    img_path: str
        path to the 3D microscopy image

    img_name: str
        name of the 3D microscopy image

    img_fmt: str
        format of the 3D microscopy image

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D FWHM of the PSF [μm]

    is_tiled: bool
        True for tiled reconstructions aligned using ZetaStitcher

    is_vec: bool
        True when pre-estimated fiber orientation vectors
        are directly provided to the pipeline

    mip_msk: bool
        apply tissue reconstruction mask (binarized MIP)

    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel

    fb_ch: int
        myelinated fibers channel

    bc_ch: int
        brain cell soma channel
    """
    # get microscopy image path and name
    img_path = cli_args.image_path
    img_name = path.basename(img_path)
    split_name = img_name.split('.')

    # check image format
    if len(split_name) == 1:
        raise ValueError('Format must be specified for input volume images!')
    else:
        img_fmt = split_name[-1]
        img_name = img_name.replace(f'.{img_fmt}', '')
        is_tiled = True if img_fmt == 'yml' else False
        is_vec = cli_args.vec

    # get image channels
    fb_ch = cli_args.fb_ch
    bc_ch = cli_args.bc_ch

    # apply tissue reconstruction mask (binarized MIP)
    msk_mip = cli_args.tissue_msk

    # apply brain cell soma mask
    msk_bc = cli_args.cell_msk

    # get Frangi filter configuration
    cfg_lbl = get_config_label(cli_args)
    img_name = f'{img_name}_{cfg_lbl}'

    # get microscopy image resolution
    px_sz, psf_fwhm = get_resolution(cli_args)

    return img_path, img_name, img_fmt, px_sz, psf_fwhm, \
        is_tiled, is_vec, msk_mip, msk_bc, fb_ch, bc_ch


def get_frangi_config(cli_args, in_img):
    """
    Get Frangi filter configuration.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

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

    Returns
    -------
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


    """
    # initialize Frangi filter configuration dictionary
    frangi_cfg = dict()

    # preprocessing configuration (in-plane smoothing)
    frangi_cfg['smooth_sd'], px_sz_iso = config_anisotropy_correction(in_img['px_sz'], in_img['psf_fwhm'])

    # Frangi filter parameters
    frangi_cfg['alpha'], frangi_cfg['beta'], frangi_cfg['gamma'] = (cli_args.alpha, cli_args.beta, cli_args.gamma)
    frangi_cfg['scales_um'] = np.array(cli_args.scales)
    frangi_cfg['scales_px'] = frangi_cfg['scales_um'] / px_sz_iso[0]

    # Frangi filter response threshold
    frangi_cfg['fb_thr'] = cli_args.fb_thr

    # fiber orientation colormap
    frangi_cfg['hsv_cmap'] = cli_args.hsv

    # forced output z-range
    z_min = int(np.floor(cli_args.z_min / np.max(in_img['px_sz'])))
    z_max = int(np.ceil(cli_args.z_max / np.max(in_img['px_sz']))) if cli_args.z_max is not None else cli_args.z_max
    frangi_cfg['z_rng'] = (z_min, z_max)

    # export all flag
    frangi_cfg['exp_all'] = cli_args.exp_all

    return frangi_cfg, px_sz_iso


def get_resolution(cli_args):
    """
    Retrieve microscopy resolution information from command line arguments.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D PSF FWHM [μm]
    """
    # create pixel and psf size arrays
    px_sz = np.array([cli_args.px_size_z, cli_args.px_size_xy, cli_args.px_size_xy])
    psf_fwhm = np.array([cli_args.psf_fwhm_z, cli_args.psf_fwhm_y, cli_args.psf_fwhm_x])

    return px_sz, psf_fwhm


def get_resource_config(cli_args):
    """
    Retrieve resource usage configuration of the Foa3D tool.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    max_ram: float
        maximum RAM available to the Frangi filter stage [B]

    jobs: int
        number of parallel jobs (threads)
        used by the Frangi filter stage
    """
    # resource parameters (convert maximum RAM to bytes)
    jobs = cli_args.jobs
    max_ram = cli_args.ram
    if max_ram is not None:
        max_ram *= 1024**3

    return max_ram, jobs


def load_microscopy_image(cli_args):
    """
    Load 3D microscopy image from TIFF, or ZetaStitcher .yml file.
    Alternatively, the processing pipeline accepts as input TIFF or NumPy
    files of fiber orientation vector data: in this case, the Frangi filter
    stage will be skipped.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    in_img: dict
        input image dictionary

        img_data: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
            3D microscopy image

        ch_ax: int
            RGB image channel axis (either 1, 3, or None for grayscale images)

        img_name: str
            name of the 3D microscopy image

        ts_msk: numpy.ndarray (dtype=bool)
            tissue reconstruction binary mask

        is_vec: bool
            vector field flag

    save_dirs: dict
        saving directories
        ('frangi': Frangi filter, 'odf': ODF analysis, 'tmp': temporary files)
    """
    # retrieve input file information
    img_path, img_name, img_fmt, px_sz, psf_fwhm, is_tiled, is_vec, \
        msk_mip, msk_bc, fb_ch, bc_ch = get_image_info(cli_args)

    # create saving directory
    save_dirs = create_save_dirs(img_path, img_name, cli_args, is_vec=is_vec)

    # import fiber orientation vector data
    tic = perf_counter()
    if is_vec:
        in_img = load_orient(img_path, img_name, img_fmt, tmp_dir=save_dirs['tmp'])

    # import raw 3D microscopy image
    else:
        in_img = load_raw(img_path, img_name, img_fmt, psf_fwhm,
                          is_tiled=is_tiled, tmp_dir=save_dirs['tmp'],
                          msk_mip=msk_mip, msk_bc=msk_bc, fb_ch=fb_ch, bc_ch=bc_ch)

    # add image pixel size to input data dictionary
    in_img['px_sz'] = px_sz

    # add image name to input data dictionary
    in_img['name'] = img_name

    # add vector field flag
    in_img['is_vec'] = is_vec

    # get microscopy image size information
    in_img = get_image_size(in_img)

    # print import time
    print_import_time(tic)

    # print volume image shape and resolution
    if not is_vec:
        print_image_info(in_img)
    else:
        print_flsh()

    return in_img, save_dirs


def load_orient(img_path, img_name, img_fmt, tmp_dir=None):
    """
    Load array of 3D fiber orientations.

    Parameters
    ----------
    img_path: str
        path to the 3D microscopy image

    img_name: str
        name of the 3D microscopy image

    img_fmt: str
        format of the 3D microscopy image

    tmp_dir: str
        temporary file directory

    Returns
    -------
    in_img: dict
        input image dictionary

            data: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
                3D microscopy image

            ts_msk: numpy.ndarray (dtype=bool)
                tissue reconstruction binary mask

            ch_ax: int
                RGB image channel axis (either 1, 3, or None for grayscale images)
    """
    # print heading
    print_flsh(color_text(0, 191, 255, "\nFiber Orientation Data Import\n"))

    # load fiber orientations
    if img_fmt == 'npy':
        vec_img = np.load(img_path, mmap_mode='r')
    elif img_fmt == 'tif' or img_fmt == 'tiff':
        vec_img = tiff.imread(img_path)
        ch_ax = detect_ch_axis(vec_img)
        if ch_ax != 3:
            vec_img = np.moveaxis(vec_img, ch_ax, -1)

        # memory-map the input TIFF image
        vec_img = create_memory_map(vec_img.shape, dtype=vec_img.dtype, name=img_name,
                                    tmp_dir=tmp_dir, arr=vec_img, mmap_mode='r')

    # check array shape
    if vec_img.ndim != 4:
        raise ValueError('Invalid 3D fiber orientation dataset (ndim != 4)!')
    else:
        print_flsh(f"Loading {img_path} orientation vector field...\n")

        # populate input image dictionary
        in_img = dict()
        in_img['data'] = vec_img
        in_img['ts_msk'] = None
        in_img['ch_ax'] = None

        return in_img


def load_raw(img_path, img_name, img_fmt, psf_fwhm,
             is_tiled=False, tmp_dir=None, msk_mip=False, msk_bc=False, fb_ch=1, bc_ch=0):
    """
    Load 3D microscopy image.

    Parameters
    ----------
    img_path: str
        path to the 3D microscopy image

    img_name: str
        name of the 3D microscopy image

    img_fmt: str
        format of the 3D microscopy image

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D FWHM of the PSF [μm]

    is_tiled: bool
        True for tiled reconstructions aligned using ZetaStitcher

    tmp_dir: str
        temporary file directory

    msk_mip: bool
        apply tissue reconstruction mask (binarized MIP)

    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel

    fb_ch: int
        myelinated fibers channel

    bc_ch: int
        brain cell soma channel

    Returns
    -------
    in_img: dict
        input image dictionary

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
    """
    # print heading
    print_flsh(color_text(0, 191, 255, "\nMicroscopy Image Import\n"))

    # load microscopy tiled reconstruction (aligned using ZetaStitcher)
    if is_tiled:
        print_flsh(f"Loading {img_path} tiled reconstruction...\n")
        img = VirtualFusedVolume(img_path)

    # load microscopy z-stack
    else:
        print_flsh(f"Loading {img_path} z-stack...\n")
        img_fmt = img_fmt.lower()
        if img_fmt == 'tif' or img_fmt == 'tiff':
            img = tiff.imread(img_path)
        else:
            raise ValueError('Unsupported image format!')

        # create image memory map
        img = create_memory_map(img.shape, dtype=img.dtype, name=img_name, tmp_dir=tmp_dir, arr=img, mmap_mode='r')

    # detect channel axis (RGB images)
    ch_ax = detect_ch_axis(img)

    # generate tissue reconstruction mask
    if msk_mip:
        dims = len(img.shape)

        # grayscale image
        if dims == 3:
            img_fbr = img
        # RGB image
        elif dims == 4:
            img_fbr = img[:, fb_ch, :, :] if ch_ax == 1 else img[..., fb_ch]
        else:
            raise ValueError('Invalid image (ndim != 3 and ndim != 4)!')

        # compute MIP (naive for loop to minimize the required RAM)
        ts_mip = np.zeros(img_fbr.shape[1:], dtype=img_fbr.dtype)
        for z in range(img_fbr.shape[0]):
            stk = np.stack((ts_mip, img_fbr[z]))
            ts_mip = np.max(stk, axis=0)

        ts_msk = create_background_mask(ts_mip, method='li', black_bg=True)
    else:
        ts_msk = None

    # populate input image dictionary
    in_img = dict()
    in_img['data'] = img
    in_img['ts_msk'] = ts_msk
    in_img['ch_ax'] = ch_ax
    in_img['fb_ch'] = fb_ch
    in_img['bc_ch'] = bc_ch
    in_img['msk_bc'] = msk_bc
    in_img['psf_fwhm'] = psf_fwhm

    return in_img
