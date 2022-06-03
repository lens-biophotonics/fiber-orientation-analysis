import argparse
import os
from argparse import RawTextHelpFormatter
from time import perf_counter

import numpy as np
import tifffile as tiff
from h5py import File
from zetastitcher import VirtualFusedVolume

from modules.cidre import correct_illumination
from modules.output import create_save_dir
from modules.preprocessing import config_anisotropy_correction
from modules.printing import (colored, print_import_time, print_resolution,
                              print_volume_shape)
from modules.utils import get_item_bytes


def cli_parser_config():
    """
    Configure the CLI argument parser object.

    Parameters
    ----------
    None

    Returns
    -------
    Configured CLI argument parser
    """
    my_parser = argparse.ArgumentParser(
        description='fiberSor: A 3D Fiber Orientation Analysis Pipeline\n' +
                    'author:     Michele Sorelli (2022)\n' +
                    'references: Frangi  et al.  (1998) ' +
                    'Multiscale vessel enhancement filtering.' +
                    ' In Medical Image Computing and' +
                    ' Computer-Assisted Intervention 1998, pp. 130-137.\n' +
                    '            Alimi   et al.  (2020) ' +
                    'Analytical and fast Fiber Orientation Distribution ' +
                    'reconstruction in 3D-Polarized Light Imaging. ' +
                    'Medical Image Analysis, 65, pp. 101760.\n\n',
        formatter_class=RawTextHelpFormatter)
    my_parser.add_argument(dest='volume_path',
                           help='path to input data volume\n' +
                                '. supported formats: .tif, .npy, ' +
                                '.yml (ZetaStitcher stitch file), ' +
                                '.h5 (4D dataset of fiber vectors)\n' +
                                '. image  axes layout: (Z, Y, X)\n' +
                                '. vector axes layout: (Z, Y, X, 3)\n\n')
    my_parser.add_argument('-a', '--alpha', type=float, default=0.001,
                           dest='alpha',
                           help='Frangi plate-like object sensitivity\n' +
                                'default: 0.001\n\n')
    my_parser.add_argument('-b', '--beta', type=float, default=1.0,
                           dest='beta',
                           help='Frangi blob-like object sensitivity\n' +
                                'default: 1.0\n\n')
    my_parser.add_argument('-g', '--gamma', type=float, default=None,
                           dest='gamma',
                           help='Frangi background score sensitivity\n' +
                                'default: automatic\n\n')
    my_parser.add_argument('-s', '--scales', nargs='+', type=float,
                           default=[1.25], dest='scales',
                           help='list of Frangi filter scales [μm]\n' +
                                'default: 1.25μm\n\n')
    my_parser.add_argument('-n', '--neuron-mask', action='store_true',
                           default=False, dest='neuron_mask',
                           help='lipofuscin-based neuronal body masking\n' +
                                'default: False\n\n')
    my_parser.add_argument('-m', '--max-slice',
                           default=100.0, dest='max_slice_size', type=float,
                           help='maximum size (in bytes) of the basic image ' +
                                'slices analyzed iteratively\n' +
                                'default: 100MB\n\n')
    my_parser.add_argument('--px-size-xy', type=float, default=0.878,
                           dest='px_size_xy',
                           help='lateral pixel size [μm]\n' +
                                'default: 0.878μm\n\n')
    my_parser.add_argument('--px-size-z', type=float, default=1.0,
                           dest='px_size_z',
                           help='longitudinal pixel size [μm]\n' +
                                'default: 1.000μm\n\n')
    my_parser.add_argument('--psf-fwhm-x', type=float, default=0.692,
                           dest='psf_fwhm_x',
                           help='PSF FWHM along the X axis [μm]\n' +
                                'default: 0.692μm\t   (MAGIC protocol; ' +
                                'Two-Photon Microscope, LENS)\n\n')
    my_parser.add_argument('--psf-fwhm-y', type=float, default=0.692,
                           dest='psf_fwhm_y',
                           help='PSF FWHM along the Y axis [μm]\n' +
                                'default: 0.692μm\t   (MAGIC protocol; ' +
                                'Two-Photon Microscope, LENS)\n\n')
    my_parser.add_argument('--psf-fwhm-z', type=float, default=2.612,
                           dest='psf_fwhm_z',
                           help='PSF FWHM along the Z axis [μm]\n' +
                                'default: 2.612μm\t   (MAGIC protocol; ' +
                                'Two-Photon Microscope, LENS)\n\n')
    my_parser.add_argument('--cp', '--cidre-path',
                           default='/mnt/NASone/michele/fiberSor/cidre',
                                   dest='cidre_path', type=str,
                           help='path to CIDRE correction models ' +
                                '(see modules.cidre correct_illumination)' +
                                '\ndefault: /mnt/NASone/michele/' +
                                'fiberSor/cidre\n\n')
    my_parser.add_argument('--cm', '--cidre-mode',
                           default=None, dest='cidre_mode', type=int,
                           help='CIDRE illumination correction ' +
                                '(options: 0, 1, 2; refer to modules.cidre)' +
                                '\ndefault: None\n\n')
    my_parser.add_argument('--ch-fiber', type=int, default=1,
                           dest='ch_fiber',
                           help='myelinated fibers channel\n' +
                                'default: 1 (green)\n\n')
    my_parser.add_argument('--ch-neuron', type=int, default=0,
                           dest='ch_neuron',
                           help='neuronal soma channel\n' +
                                'default: 0 (red)\n\n')
    my_parser.add_argument('--z-min', type=int, default=0,
                           dest='z_min',
                           help='forced minimum output z-depth\n\n')
    my_parser.add_argument('--z-max', type=int, default=None,
                           dest='z_max',
                           help='forced maximum output z-depth\n\n')
    my_parser.add_argument('--or', '--odf-res', nargs='+', type=float,
                           default=[], dest='odf_res',
                           help='side of the ODF super-voxels [μm]' +
                                '\ndefault: [] (do not estimate ODFs)\n\n')
    my_parser.add_argument('--oo', '--odf-order', type=int,
                           default=6, dest='odf_order',
                           help='spherical harmonics series expansion order ' +
                                '(even number between 2 and 10)\n' +
                                'default: 6\n\n')

    return my_parser


def load_input_volume(parser, cidre_mode=None):
    """
    Load TPFM volume from TIFF, NumPy or ZetaStitcher .yml file.
    Apply CIDRE-based illumination correction if required.
    Alternatively, the processing pipeline accepts .h5 datasets of
    fiber orientation vectors as input: in this case, the
    Frangi filtering stage will be skipped.

    Parameters
    ----------
    parser: argument parser object

    cidre_mode: int
        correction mode flag

    Returns
    -------
    volume: ndarray or HDF5 dataset
            (shape=(Z,C,Y,X) for .yml stitch files,
             shape=(Z,Y,X,3) for .h5 datasets,
             shape=(Z,Y,X,C) otherwise)
        TPFM image volume or fiber orientation vector dataset

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher
    """
    # print heading
    print(colored(0, 191, 255, "\n  Volume Import\n"))

    # retrieve volume path and name
    args = parser.parse_args()
    volume_path = args.volume_path
    volume_fname = os.path.basename(volume_path)
    split_fname = volume_fname.split('.')
    volume_name = split_fname[0]
    mosaic = False
    vector = False

    # check input volume format (ZetaStitcher .yml file)
    if len(split_fname) == 1:
        raise ValueError('  Format must be specified for input image volumes!')
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

    # microscopy image volume
    else:
        # CIDRE illumination correction
        cidre_path = args.cidre_path
        cidre_mode = args.cidre_mode
        if cidre_mode is not None:
            volume_path = correct_illumination(volume_path, models=cidre_path,
                                               mosaic=mosaic, mode=cidre_mode)

        # load TPFM tiled reconstruction (aligned using ZetaStitcher)
        if mosaic:
            print("  Loading " + volume_name + " tiled reconstruction...\n")
            volume = VirtualFusedVolume(volume_path)
        # load TPFM z-stack
        else:
            print("  Loading " + volume_fname + " z-stack...\n")
            volume_format = volume_format.lower()
            if volume_format == 'npy':
                volume = np.load(volume_path)
            elif volume_format == 'tif' or volume_format == 'tiff':
                volume = tiff.imread(volume_path)

    # print import time
    print_import_time(tic)

    # print volume shape
    print_volume_shape(volume, parser, mosaic)

    return volume, mosaic, vector


def get_volume_info(volume, px_size, mosaic=False):
    """
    Get information on the input microscopy volume.

    Parameters
    ----------
    volume: ndarray
        TPFM image array

    px_size: ndarray (shape=(3,), dtype=float)
        original TPFM spatial resolution in [μm]

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    volume_shape: ndarray (shape=(3,), dtype=int)
        volume shape [px]

    volume_shape_um: ndarray (shape=(3,), dtype=float)
        volume shape [μm]

    volume_item_size: int
        array item size (in bytes)
    """
    # adapt channel axis
    if mosaic:
        channel_axis = 1
    else:
        channel_axis = -1

    # get microscopy volume info
    volume_shape = np.asarray(volume.shape)
    volume_shape = np.delete(volume_shape, channel_axis)
    volume_shape_um = np.multiply(volume_shape, px_size)
    volume_item_size = get_item_bytes(volume)

    return volume_shape, volume_shape_um, volume_item_size


def load_pipeline_config(parser):
    """
    Retrieve Frangi filter pipeline configuration.

    Parameters
    ----------
    parser: argument parser object

    Returns
    -------
    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    smooth_sigma: ndarray
        3D standard deviation of low-pass Gaussian filter [px]

    px_size: ndarray
        original TPFM spatial resolution in [μm]

    px_size_iso: ndarray
        adjusted isotropic TPFM spatial resolution in [μm]

    odf_scales_um: list (dtype: float)
        list of fiber ODF resolution values (super-voxel sides in [μm])

    odf_order: int
        order of the spherical harmonics series expansion

    z_min: int
        minimum output z-depth in [px]

    z_max: int
        maximum output z-depth in [px]

    ch_neuron: int
        neuronal bodies channel

    ch_fiber: int
        myelinated fibers channel

    max_slice_size: float
        maximum memory size (in bytes) of the basic slices analyzed iteratively

    lpf_soma_mask: bool
        neuronal body masking flag

    save_dir: str
        saving directory path

    volume_name: str
        name of the input TPFM volume
    """
    # volume filename
    args = parser.parse_args()
    volume_path = args.volume_path
    volume_fname = os.path.basename(volume_path)
    volume_name = volume_fname.split('.')[0]

    # Frangi filter parameters
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    scales_um = args.scales
    if type(scales_um) is not list:
        scales_um = [scales_um]
    volume_name = 'a'+str(alpha)+'_b'+str(beta)+'_g'+str(gamma)+'_'+volume_name

    # pipeline flags
    lpf_soma_mask = args.neuron_mask
    ch_neuron = args.ch_neuron
    ch_fiber = args.ch_fiber
    max_slice_size = args.max_slice_size

    # ODF analysis
    odf_scales_um = args.odf_res
    odf_order = args.odf_order

    # TPFM pixel size and PSF FWHM
    px_size, psf_fwhm = load_microscope_config(parser)

    # forced output z-range
    z_min = args.z_min
    z_max = args.z_max
    z_min = int(np.floor(z_min / px_size[0]))
    if z_max is not None:
        z_max = int(np.ceil(z_max / px_size[0]))

    # preprocessing configuration
    smooth_sigma, px_size_iso = config_anisotropy_correction(px_size, psf_fwhm)

    # create saving directory
    save_dir = create_save_dir(volume_path)

    return alpha, beta, gamma, scales_um, smooth_sigma, px_size, px_size_iso, \
        odf_scales_um, odf_order, z_min, z_max, ch_neuron, ch_fiber, \
        max_slice_size, lpf_soma_mask, save_dir, volume_name


def load_microscope_config(parser):
    """
    Retrieve TPFM microscope parameters from command line.

    Parameters
    ----------
    parser: argument parser object

    Returns
    -------
    px_size: ndarray (shape=(3,), dtype=float)
        original TPFM pixel size in [μm]

    psf_fwhm: ndarray (shape=(3,), dtype=float)
        3D PSF FWHM in [μm]
    """
    # retrieve TPFM properties
    args = parser.parse_args()

    # pixel size
    px_size_z = args.px_size_z
    px_size_xy = args.px_size_xy

    # psf
    psf_fwhm_z = args.psf_fwhm_z
    psf_fwhm_y = args.psf_fwhm_y
    psf_fwhm_x = args.psf_fwhm_x

    px_size = np.array([px_size_z, px_size_xy, px_size_xy])
    psf_fwhm = np.array([psf_fwhm_z, psf_fwhm_y, psf_fwhm_x])

    # print TPFM resolution info
    print_resolution(px_size, psf_fwhm)

    return px_size, psf_fwhm
