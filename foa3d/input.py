import argparse
import os
from time import perf_counter

import numpy as np
import tifffile as tiff

try:
    from zetastitcher import VirtualFusedVolume
except ImportError:
    pass

import tempfile
from os import path

from foa3d.output import create_save_dirs
from foa3d.preprocessing import config_anisotropy_correction
from foa3d.printing import (color_text, print_image_shape, print_import_time,
                            print_resolution)
from foa3d.utils import create_memory_map, get_item_bytes, get_output_prefix


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def cli_parser():
    """
    Parse command line arguments.

    Parameters
    ----------
    None

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
                    'Medical Image Analysis, 65, pp. 101760.\n\n',
        formatter_class=CustomFormatter)
    cli_parser.add_argument(dest='image_path',
                            help='path to input microscopy volume image\n'
                                 '* supported formats: .tif, .npy, .yml (ZetaStitcher stitch file), '
                                 '.h5 (4D dataset of fiber vectors)\n'
                                 '* image  axes order: (Z, Y, X)\n'
                                 '* vector axes order: (Z, Y, X, 3)')
    cli_parser.add_argument('-a', '--alpha', type=float, default=0.001,
                            help='Frangi plate-like object sensitivity')
    cli_parser.add_argument('-b', '--beta', type=float, default=1.0,
                            help='Frangi blob-like object sensitivity')
    cli_parser.add_argument('-g', '--gamma', type=float, default=None,
                            help='Frangi background score sensitivity')
    cli_parser.add_argument('-s', '--scales', nargs='+', type=float, default=[1.25],
                            help='list of Frangi filter scales [μm]')
    cli_parser.add_argument('-n', '--neuron-mask', action='store_true', default=False,
                            help='lipofuscin-based neuronal body masking')
    cli_parser.add_argument('-j', '--jobs-prc', type=float, default=80.0,
                            help='maximum parallel jobs relative to the number of available CPU cores (percentage)')
    cli_parser.add_argument('-r', '--ram', type=float, default=None,
                            help='maximum RAM available to the Frangi filtering stage [GB] (default: use all)')
    cli_parser.add_argument('--px-size-xy', type=float, default=0.878, help='lateral pixel size [μm]')
    cli_parser.add_argument('--px-size-z', type=float, default=1.0, help='longitudinal pixel size [μm]')
    cli_parser.add_argument('--psf-fwhm-x', type=float, default=0.692, help='PSF FWHM along the X axis [μm]')
    cli_parser.add_argument('--psf-fwhm-y', type=float, default=0.692, help='PSF FWHM along the Y axis [μm]')
    cli_parser.add_argument('--psf-fwhm-z', type=float, default=2.612, help='PSF FWHM along the Z axis [μm]\n')
    cli_parser.add_argument('--ch-fiber', type=int, default=1, help='myelinated fibers channel')
    cli_parser.add_argument('--ch-neuron', type=int, default=0, help='neuronal soma channel')
    cli_parser.add_argument('--z-min', type=float, default=0, help='forced minimum output z-depth [μm]')
    cli_parser.add_argument('--z-max', type=float, default=None, help='forced maximum output z-depth [μm]')
    cli_parser.add_argument('--odf-res', nargs='+', type=float, help='side of the ODF super-voxels [μm]')
    cli_parser.add_argument('--odf-deg', type=int, default=6,
                            help='degrees of the spherical harmonics series expansion (even number between 2 and 10)')

    # parse arguments
    cli_args = cli_parser.parse_args()

    return cli_args


def get_image_info(img, px_size, ch_fiber, mosaic=False):
    """
    Get information on the input microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    ch_fiber: int
        myelinated fibers channel

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    img_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    img_item_size: int
        array item size (in bytes)
    """
    # adapt channel axis
    img_shape = np.asarray(img.shape)
    ndim = len(img_shape)
    if ndim == 4:
        if mosaic:
            ch_axis = 1
        else:
            ch_axis = -1
    elif ndim == 3:
        ch_fiber = None

    # get info on microscopy volume image
    if ch_axis is not None:
        img_shape = np.delete(img_shape, ch_axis)
    img_shape_um = np.multiply(img_shape, px_size)
    img_item_size = get_item_bytes(img)

    return img_shape, img_shape_um, img_item_size, ch_fiber


def get_pipeline_config(cli_args, vector, img_name):
    """
    Retrieve the Foa3D pipeline configuration.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    vector: bool
        True for fiber orientation vector data

    img_name: str
        name of the input volume image

    Returns
    -------
    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    odf_scales_um: list (dtype: float)
        list of fiber ODF resolution values (super-voxel sides [μm])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    z_min: int
        minimum output z-depth [px]

    z_max: int
        maximum output z-depth [px]

    ch_neuron: int
        neuronal bodies channel

    ch_fiber: int
        myelinated fibers channel

    lpf_soma_mask: bool
        neuronal body masking

    max_ram_mb: float
        maximum RAM available to the Frangi filtering stage [MB]

    jobs_to_cores: float
        max number of jobs relative to the available CPU cores
        (default: 80%)

    img_name: str
        microscopy image filename
    """
    # Frangi filter parameters
    alpha = cli_args.alpha
    beta = cli_args.beta
    gamma = cli_args.gamma
    scales_um = cli_args.scales
    if type(scales_um) is not list:
        scales_um = [scales_um]

    # add pipeline configuration prefix to input volume name
    if not vector:
        pfx = get_output_prefix(scales_um, alpha, beta, gamma)
        img_name = pfx + 'img' + img_name

    # pipeline flags
    lpf_soma_mask = cli_args.neuron_mask
    ch_neuron = cli_args.ch_neuron
    ch_fiber = cli_args.ch_fiber
    max_ram = cli_args.ram
    max_ram_mb = None if max_ram is None else max_ram * 1000
    jobs_prc = cli_args.jobs_prc
    jobs_to_cores = 1 if jobs_prc >= 100 else 0.01 * jobs_prc

    # ODF analysis
    odf_scales_um = cli_args.odf_res
    odf_degrees = cli_args.odf_deg

    # TPFM pixel size and PSF FWHM
    px_size, psf_fwhm = get_resolution(cli_args, vector)

    # forced output z-range
    z_min = cli_args.z_min
    z_max = cli_args.z_max
    z_min = int(np.floor(z_min / px_size[0]))
    if z_max is not None:
        z_max = int(np.ceil(z_max / px_size[0]))

    # preprocessing configuration
    smooth_sigma, px_size_iso = config_anisotropy_correction(px_size, psf_fwhm, vector)

    return alpha, beta, gamma, scales_um, smooth_sigma, px_size, px_size_iso, odf_scales_um, odf_degrees, \
        z_min, z_max, ch_neuron, ch_fiber, lpf_soma_mask, max_ram_mb, jobs_to_cores, img_name


def get_resolution(cli_args, vector):
    """
    Retrieve microscopy resolution from command line arguments.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D PSF FWHM [μm]
    """
    # pixel size
    px_size_z = cli_args.px_size_z
    px_size_xy = cli_args.px_size_xy

    # psf
    psf_fwhm_z = cli_args.psf_fwhm_z
    psf_fwhm_y = cli_args.psf_fwhm_y
    psf_fwhm_x = cli_args.psf_fwhm_x

    px_size = np.array([px_size_z, px_size_xy, px_size_xy])
    psf_fwhm = np.array([psf_fwhm_z, psf_fwhm_y, psf_fwhm_x])

    # print TPFM resolution info
    if not vector:
        print_resolution(px_size, psf_fwhm)

    return px_size, psf_fwhm


def load_microscopy_image(cli_args):
    """
    Load microscopy volume image from TIFF, NumPy or ZetaStitcher .yml file.
    Alternatively, the processing pipeline accepts as input .h5 datasets of
    fiber orientation vectors: in this case, the Frangi filter stage will be
    skipped.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    img: NumPy memory map
        microscopy volume image or dataset of fiber orientation vectors

    mosaic: bool
        True for tiled microscopy reconstructions aligned using ZetaStitcher

    skip_frangi: bool
        True when fiber orientation vectors are provided as input
        to the pipeline

    cli_args: see ArgumentParser.parse_args
        updated namespace of command line arguments

    save_subdirs: list (dtype=str)
        saving subdirectory string paths

    tmp_dir: str
        temporary file directory

    img_name: str
        microscopy image filename
    """
    # retrieve volume path and name
    img_path = cli_args.image_path
    img_fname = os.path.basename(img_path)
    split_fname = img_fname.split('.')
    img_name = img_fname.replace('.' + split_fname[-1], '')
    mosaic = False
    skip_frangi = False
    skip_odf = cli_args.odf_res is None

    # create temporary file directory
    tmp_dir = tempfile.mkdtemp()

    # check input volume image format (ZetaStitcher .yml file)
    if len(split_fname) == 1:
        raise ValueError('Format must be specified for input volume images!')
    else:
        img_fmt = split_fname[-1]
        if img_fmt == 'yml':
            mosaic = True

    # import start time
    tic = perf_counter()

    # fiber orientation vectors dataset
    if img_fmt == 'npy':

        # print heading
        print(color_text(0, 191, 255, "\nFiber Orientation Data Import\n"))

        # load fiber orientation data
        img = np.load(img_path, mmap_mode='r')

        # check dimensions
        if img.ndim != 4:
            raise ValueError('Invalid fiber orientation dataset (ndim != 4)')
        else:
            skip_frangi = True
            print("Loading " + img_name + " orientation dataset...\n")

    # microscopy volume image
    else:
        # print heading
        print(color_text(0, 191, 255, "\nMicroscopy Volume Image Import\n"))

        # load microscopy tiled reconstruction (aligned using ZetaStitcher)
        if mosaic:
            print("Loading " + img_name + " tiled reconstruction...\n")
            img = VirtualFusedVolume(img_path)

        # load microscopy z-stack
        else:
            print("Loading " + img_fname + " z-stack...\n")
            img_fmt = img_fmt.lower()
            if img_fmt == 'npy':
                img = np.load(img_path)
            elif img_fmt == 'tif' or img_fmt == 'tiff':
                img = tiff.imread(img_path)

        # grey channel fiber image
        if len(img.shape) == 3:
            cli_args.neuron_mask = False

        # create image memory map
        mmap_path = path.join(tmp_dir, 'tmp_' + img_name + '.mmap')
        img = create_memory_map(mmap_path, img.shape, dtype=img.dtype, arr=img[:], mmap_mode='r')

    # print import time
    print_import_time(tic)

    # create saving directory
    save_subdirs = create_save_dirs(img_path, img_name, skip_frangi=skip_frangi, skip_odf=skip_odf)

    # print volume image shape
    if not skip_frangi:
        print_image_shape(cli_args, img, mosaic)
    else:
        print()

    return img, mosaic, skip_frangi, cli_args, save_subdirs, tmp_dir, img_name
