import argparse
import tempfile
from time import perf_counter

import h5py
import numpy as np
import tifffile as tiff

try:
    from zetastitcher import VirtualFusedVolume
except ImportError:
    pass

from os import path

from foa3d.output import create_save_dirs
from foa3d.preprocessing import config_anisotropy_correction
from foa3d.printing import (color_text, print_image_shape, print_import_time,
                            print_native_res)
from foa3d.utils import (create_background_mask, create_memory_map,
                         get_item_bytes, get_output_prefix)


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
                            help='path to input 3D microscopy image or to 4D array of fiber orientation vectors\n'
                                 '* supported formats: .tif (image), '
                                 '.npy (image or fiber vectors), .yml (ZetaStitcher stitch file)\n'
                                 '* image  axes order: (Z, Y, X)\n'
                                 '* vector axes order: (Z, Y, X, C)')
    cli_parser.add_argument('-a', '--alpha', type=float, default=0.001,
                            help='Frangi plate-like object sensitivity')
    cli_parser.add_argument('-b', '--beta', type=float, default=1.0,
                            help='Frangi blob-like object sensitivity')
    cli_parser.add_argument('-g', '--gamma', type=float, default=None,
                            help='Frangi background score sensitivity')
    cli_parser.add_argument('-s', '--scales', nargs='+', type=float, default=[1.25],
                            help='list of Frangi filter scales [μm]')
    cli_parser.add_argument('-j', '--jobs', type=int, default=None,
                            help='number of parallel threads used by the Frangi filtering stage: '
                                 'use one thread per logical core if None')
    cli_parser.add_argument('-r', '--ram', type=float, default=None,
                            help='maximum RAM available to the Frangi filtering stage [GB]: use all if None')
    cli_parser.add_argument('-m', '--mmap', action='store_true', default=False,
                            help='create a memory-mapped array of the 3D microscopy image')
    cli_parser.add_argument('--px-size-xy', type=float, default=0.878, help='lateral pixel size [μm]')
    cli_parser.add_argument('--px-size-z', type=float, default=1.0, help='longitudinal pixel size [μm]')
    cli_parser.add_argument('--psf-fwhm-x', type=float, default=0.692, help='PSF FWHM along the X axis [μm]')
    cli_parser.add_argument('--psf-fwhm-y', type=float, default=0.692, help='PSF FWHM along the Y axis [μm]')
    cli_parser.add_argument('--psf-fwhm-z', type=float, default=2.612, help='PSF FWHM along the Z axis [μm]')
    cli_parser.add_argument('--fb-ch', type=int, default=1, help='myelinated fibers channel')
    cli_parser.add_argument('--bc-ch', type=int, default=0, help='neuronal bodies channel')
    cli_parser.add_argument('--z-min', type=float, default=0, help='forced minimum output z-depth [μm]')
    cli_parser.add_argument('--z-max', type=float, default=None, help='forced maximum output z-depth [μm]')
    cli_parser.add_argument('--hsv', action='store_true', default=False,
                            help='toggle HSV colormap for 3D fiber orientations')
    cli_parser.add_argument('--odf-res', nargs='+', type=float, help='side of the fiber ODF super-voxels: '
                                                                     'do not generate ODFs if None [μm]')
    cli_parser.add_argument('--odf-deg', type=int, default=6,
                            help='degrees of the spherical harmonics series expansion (even number between 2 and 10)')
    cli_parser.add_argument('-o', '--out', type=str, default=None,
                            help='output directory')
    cli_parser.add_argument('-c', '--cell-msk', action='store_true', default=False,
                            help='toggle optional neuronal body masking')
    cli_parser.add_argument('-t', '--tissue-msk', action='store_true', default=False,
                            help='apply tissue reconstruction mask (binarized MIP)')

    # parse arguments
    cli_args = cli_parser.parse_args()

    return cli_args


def get_image_info(img, px_sz, msk_bc, fb_ch, ch_ax=None, is_tiled=False):
    """
    Get information on the input 3D microscopy image.

    Parameters
    ----------
    img: numpy.ndarray
        3D microscopy image

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel

    fb_ch: int
        myelinated fibers channel

    ch_ax: int
        channel axis

    is_tiled: bool
        True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    img_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    img_item_sz: int
        image item size [B]

    fb_ch: int
        myelinated fibers channel

    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel
    """

    # adapt channel axis
    img_shp = np.asarray(img.shape)
    ndim = len(img_shp)
    if ndim == 4:
        ch_ax = 1 if is_tiled else -1
    elif ndim == 3:
        fb_ch = None
        msk_bc = False

    # get info on 3D microscopy image
    if ch_ax is not None:
        img_shp = np.delete(img_shp, ch_ax)
    img_shp_um = np.multiply(img_shp, px_sz)
    img_item_sz = get_item_bytes(img)

    return img_shp, img_shp_um, img_item_sz, fb_ch, msk_bc


def get_file_info(cli_args):
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

    is_tiled: bool
        True for tiled reconstructions aligned using ZetaStitcher

    is_mmap: bool
        create a memory-mapped array of the 3D microscopy image,
        increasing the parallel processing performance
        (the image will be preliminarily loaded to RAM)

    mip_msk: bool
        apply tissue reconstruction mask (binarized MIP)

    fb_ch: int
        myelinated fibers channel
    """

    # get microscopy image path and name
    is_mmap = cli_args.mmap
    img_path = cli_args.image_path
    img_name = path.basename(img_path)
    split_name = img_name.split('.')

    # check image format
    if len(split_name) == 1:
        raise ValueError('Format must be specified for input volume images!')
    else:
        img_fmt = split_name[-1]
        img_name = img_name.replace('.{}'.format(img_fmt), '')
        is_tiled = True if img_fmt == 'yml' else False

    # apply tissue reconstruction mask (binarized MIP)
    msk_mip = cli_args.tissue_msk
    fb_ch = cli_args.fb_ch

    return img_path, img_name, img_fmt, is_tiled, is_mmap, msk_mip, fb_ch


def get_frangi_config(cli_args, img_name):
    """
    Get Frangi filter configuration.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    img_name: str
        name of the 3D microscopy image

    Returns
    -------
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

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [px]

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_sz_iso: int
        isotropic pixel size [μm]

    z_rng: int
        output z-range in [px]

    bc_ch: int
        neuronal bodies channel

    fb_ch: int
        myelinated fibers channel

    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel

    hsv_vec_cmap: bool

    out_name: str
        output file name
    """

    # microscopy image pixel and PSF size
    px_sz, psf_fwhm = get_resolution(cli_args)

    # preprocessing configuration (in-plane smoothing)
    smooth_sigma, px_sz_iso = config_anisotropy_correction(px_sz, psf_fwhm)

    # Frangi filter parameters
    alpha, beta, gamma = (cli_args.alpha, cli_args.beta, cli_args.gamma)
    scales_um = np.array(cli_args.scales)
    scales_px = scales_um / px_sz_iso[0]

    # image channels
    fb_ch = cli_args.fb_ch
    bc_ch = cli_args.bc_ch
    msk_bc = cli_args.cell_msk

    # fiber orientation colormap
    hsv_vec_cmap = cli_args.hsv

    # forced output z-range
    z_min = int(np.floor(cli_args.z_min / px_sz[0]))
    z_max = int(np.ceil(cli_args.z_max / px_sz[0])) if cli_args.z_max is not None else cli_args.z_max
    z_rng = (z_min, z_max)

    # add Frangi filter configuration prefix to output filenames
    pfx = get_output_prefix(scales_um, alpha, beta, gamma)
    out_name = '{}img{}'.format(pfx, img_name)

    return alpha, beta, gamma, scales_px, scales_um, smooth_sigma, \
        px_sz, px_sz_iso, z_rng, bc_ch, fb_ch, msk_bc, hsv_vec_cmap, out_name


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

    # print resolution info
    print_native_res(px_sz, psf_fwhm)

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
    Load 3D microscopy image from TIFF, NumPy or ZetaStitcher .yml file.
    Alternatively, the processing pipeline accepts as input NumPy or HDF5
    files of fiber orientation vector data: in this case, the Frangi filter
    stage will be skipped.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    img: numpy.ndarray or NumPy memory-map object
        3D microscopy image or array of fiber orientation vectors

    tissue_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    is_tiled: bool
        True for tiled microscopy reconstructions aligned using ZetaStitcher

    is_fiber: bool
        True when pre-estimated fiber orientation vectors
        are directly provided to the pipeline

    save_dir: list (dtype=str)
        saving subdirectory string paths

    tmp_dir: str
        temporary file directory

    img_name: str
        microscopy image filename
    """

    # create temporary directory
    tmp_dir = tempfile.mkdtemp()

    # retrieve input file information
    img_path, img_name, img_fmt, is_tiled, is_mmap, msk_mip, fb_ch = get_file_info(cli_args)

    # import fiber orientation vector data
    tic = perf_counter()
    if img_fmt == 'npy' or img_fmt == 'h5':
        img, is_fiber = load_orient(img_path, img_name, img_fmt)
        tissue_msk = None

    # import raw 3D microscopy image
    else:
        img, tissue_msk, is_fiber = load_raw(img_path, img_name, img_fmt, is_tiled=is_tiled, is_mmap=is_mmap,
                                             tmp_dir=tmp_dir, msk_mip=msk_mip, fb_ch=fb_ch)

    # print import time
    print_import_time(tic)

    # print volume image shape
    print_image_shape(cli_args, img, is_tiled) if not is_fiber else print()

    # create saving directory
    save_dir = create_save_dirs(img_path, img_name, cli_args, is_fiber=is_fiber)

    return img, tissue_msk, is_tiled, is_fiber, save_dir, tmp_dir, img_name


def load_orient(img_path, img_name, img_fmt):
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

    Returns
    -------
    img: numpy.ndarray
        3D fiber orientation vectors

    is_fiber: bool
        True when pre-estimated fiber orientation vectors
        are directly provided to the pipeline
    """

    # print heading
    print(color_text(0, 191, 255, "\nFiber Orientation Data Import\n"))

    # load fiber orientations
    if img_fmt == 'npy':
        img = np.load(img_path, mmap_mode='r')
    else:
        img_file = h5py.File(img_path, 'r')
        img = img_file.get(img_file.keys()[0])

    # check array shape
    if img.ndim != 4:
        raise ValueError('Invalid 3D fiber orientation dataset (ndim != 4)!')
    else:
        is_fiber = True
        print("Loading {} orientation dataset...\n".format(img_name))

        return img, is_fiber


def load_raw(img_path, img_name, img_fmt, is_tiled=False, is_mmap=False, tmp_dir=None, msk_mip=False, fb_ch=1):
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

    is_tiled: bool
        True for tiled reconstructions aligned using ZetaStitcher

    is_mmap: bool
        create a memory-mapped array of the 3D microscopy image,
        increasing the parallel processing performance
        (the image will be preliminarily loaded to RAM)

    tmp_dir: str
        temporary file directory

    mip_msk: bool
        apply tissue reconstruction mask (binarized MIP)

    fb_ch: int
        myelinated fibers channel

    Returns
    -------
    img: numpy.ndarray or NumPy memory-map object
        3D microscopy image

    tissue_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    is_fiber: bool
        True when pre-estimated fiber orientation vectors
        are directly provided to the pipeline
    """

    # print heading
    print(color_text(0, 191, 255, "\nMicroscopy Image Import\n"))

    # load microscopy tiled reconstruction (aligned using ZetaStitcher)
    if is_tiled:
        print("Loading {} tiled reconstruction...\n".format(img_name))
        img = VirtualFusedVolume(img_path)

    # load microscopy z-stack
    else:
        print("Loading {} z-stack...\n".format(img_name))
        img_fmt = img_fmt.lower()
        if img_fmt == 'npy':
            img = np.load(img_path)
        elif img_fmt == 'tif' or img_fmt == 'tiff':
            img = tiff.imread(img_path)
        else:
            raise ValueError('Unsupported image format!')

    # create image memory map
    if is_mmap:
        img = create_memory_map(img.shape, dtype=img.dtype, name=img_name, tmp_dir=tmp_dir, arr=img[:], mmap_mode='r')

    # compute tissue reconstruction mask (binarized MIP)
    if msk_mip:
        dims = len(img.shape)
        if dims == 3:
            tissue_mip = np.max(img[:], axis=0)
        elif dims == 4:
            img_mye = img[:, fb_ch, :, :] if is_tiled else img[..., fb_ch]
            tissue_mip = np.max(img_mye, axis=0)

        tissue_msk = create_background_mask(tissue_mip, method='li', black_bg=True)

    else:
        tissue_msk = None

    # raw image
    is_fiber = False

    return img, tissue_msk, is_fiber
