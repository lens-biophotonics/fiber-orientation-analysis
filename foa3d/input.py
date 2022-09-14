import argparse
import os
from time import perf_counter

import numpy as np
import tifffile as tiff
from h5py import File
from zetastitcher import VirtualFusedVolume

from foa3d.output import create_save_dir
from foa3d.preprocessing import config_anisotropy_correction
from foa3d.printing import (color_text, print_import_time, print_resolution,
                            print_volume_shape)
from foa3d.utils import get_item_bytes


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
    cli_parser.add_argument(dest='volume_path',
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
    cli_parser.add_argument('-m', '--max-slice-size', default=100.0, type=float,
                            help='maximum size (in MegaBytes) of the basic image slices analyzed iteratively')
    cli_parser.add_argument('--px-size-xy', type=float, default=0.878, help='lateral pixel size [μm]')
    cli_parser.add_argument('--px-size-z', type=float, default=1.0, help='longitudinal pixel size [μm]')
    cli_parser.add_argument('--psf-fwhm-x', type=float, default=0.692, help='PSF FWHM along the X axis [μm]')
    cli_parser.add_argument('--psf-fwhm-y', type=float, default=0.692, help='PSF FWHM along the Y axis [μm]')
    cli_parser.add_argument('--psf-fwhm-z', type=float, default=2.612, help='PSF FWHM along the Z axis [μm]\n')
    cli_parser.add_argument('--ch-fiber', type=int, default=1, help='myelinated fibers channel')
    cli_parser.add_argument('--ch-neuron', type=int, default=0, help='neuronal soma channel')
    cli_parser.add_argument('--z-min', type=int, default=0, help='forced minimum output z-depth')
    cli_parser.add_argument('--z-max', type=int, default=None, help='forced maximum output z-depth')
    cli_parser.add_argument('--odf-res', nargs='+', type=float, help='side of the ODF super-voxels [μm]')
    cli_parser.add_argument('--odf-deg', type=int, default=6,
                            help='degrees of the spherical harmonics series expansion (even number between 2 and 10)')

    # parse arguments
    cli_args = cli_parser.parse_args()

    return cli_args


def get_pipeline_config(cli_args):
    """
    Retrieve the Foa3D pipeline configuration.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

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

    max_slice_size: float
        maximum memory size (in bytes) of the basic image slices
        analyzed iteratively

    lpf_soma_mask: bool
        neuronal body masking

    save_dir: str
        saving directory path

    volume_name: str
        name of the input volume image
    """
    # volume filename
    volume_path = cli_args.volume_path
    volume_fname = os.path.basename(volume_path)
    volume_name = volume_fname.split('.')[0]

    # Frangi filter parameters
    alpha = cli_args.alpha
    beta = cli_args.beta
    gamma = cli_args.gamma
    scales_um = cli_args.scales
    if type(scales_um) is not list:
        scales_um = [scales_um]
    volume_name = 'a' + str(alpha) + '_b' + str(beta) + '_g' + str(gamma) + '_' + volume_name

    # pipeline flags
    lpf_soma_mask = cli_args.neuron_mask
    ch_neuron = cli_args.ch_neuron
    ch_fiber = cli_args.ch_fiber
    max_slice_size = cli_args.max_slice_size

    # ODF analysis
    odf_scales_um = cli_args.odf_res
    odf_degrees = cli_args.odf_deg

    # TPFM pixel size and PSF FWHM
    px_size, psf_fwhm = get_resolution(cli_args)

    # forced output z-range
    z_min = cli_args.z_min
    z_max = cli_args.z_max
    z_min = int(np.floor(z_min / px_size[0]))
    if z_max is not None:
        z_max = int(np.ceil(z_max / px_size[0]))

    # preprocessing configuration
    smooth_sigma, px_size_iso = config_anisotropy_correction(px_size, psf_fwhm)

    # create saving directory
    save_dir = create_save_dir(volume_path)

    return alpha, beta, gamma, scales_um, smooth_sigma, px_size, px_size_iso, odf_scales_um, odf_degrees, \
        z_min, z_max, ch_neuron, ch_fiber, max_slice_size, lpf_soma_mask, save_dir, volume_name


def get_resolution(cli_args):
    """
    Retrieve microscopy resolution from command line arguments.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    px_size: ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: ndarray (shape=(3,), dtype=float)
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
    print_resolution(px_size, psf_fwhm)

    return px_size, psf_fwhm


def get_volume_info(volume, px_size, mosaic=False):
    """
    Get information on the input microscopy volume image.

    Parameters
    ----------
    volume: numpy.ndarray
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    volume_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    volume_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    volume_item_size: int
        array item size (in bytes)
    """
    # adapt channel axis
    if mosaic:
        channel_axis = 1
    else:
        channel_axis = -1

    # get info on microscopy volume image
    volume_shape = np.asarray(volume.shape)
    volume_shape = np.delete(volume_shape, channel_axis)
    volume_shape_um = np.multiply(volume_shape, px_size)
    volume_item_size = get_item_bytes(volume)

    return volume_shape, volume_shape_um, volume_item_size


def load_volume(cli_args):
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
    volume: numpy.ndarray or HDF5 dataset
        microscopy volume image or dataset of fiber orientation vectors

    mosaic: bool
        True for tiled microscopy reconstructions aligned using ZetaStitcher
    """
    # print heading
    print(color_text(0, 191, 255, "\n  Microscopy Volume Image Import\n"))

    # retrieve volume path and name
    volume_path = cli_args.volume_path
    volume_fname = os.path.basename(volume_path)
    split_fname = volume_fname.split('.')
    volume_name = split_fname[0]
    mosaic = False
    vector = False

    # check input volume format (ZetaStitcher .yml file)
    if len(split_fname) == 1:
        raise ValueError('  Format must be specified for input volume images!')
    else:
        volume_format = split_fname[-1]
        if volume_format == 'yml':
            mosaic = True

    # import start time
    tic = perf_counter()

    # fiber orientation vectors dataset
    if volume_format == 'h5':
        hf = File(volume_path, 'r')
        volume = hf.get('chunked')

        # check dimensions
        if volume.ndim != 4:
            raise ValueError('  Invalid fiber orientation dataset (ndim != 4)')
        else:
            vector = True
            print("  Loading " + volume_name + " orientation dataset...\n")

    # microscopy volume image
    else:
        # load tiled reconstruction (aligned using ZetaStitcher)
        if mosaic:
            print("  Loading " + volume_name + " tiled reconstruction...\n")
            volume = VirtualFusedVolume(volume_path)
        # load z-stack
        else:
            print("  Loading " + volume_fname + " z-stack...\n")
            volume_format = volume_format.lower()
            if volume_format == 'npy':
                volume = np.load(volume_path)
            elif volume_format == 'tif' or volume_format == 'tiff':
                volume = tiff.imread(volume_path)

    # print import time
    print_import_time(tic)

    # print volume image shape
    print_volume_shape(cli_args, volume, mosaic)

    return volume, mosaic, vector
