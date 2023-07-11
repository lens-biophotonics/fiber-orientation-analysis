from datetime import datetime
from os import mkdir, path

import nibabel as nib
import numpy as np
import tifffile as tiff


def create_save_dirs(img_path, img_name, skip_frangi=False, skip_odf=False):
    """
    Create saving directory.

    Parameters
    ----------
    img_path: str
        path to input microscopy volume image

    img_name: str
        name of the input volume image

    skip_frangi: bool
        True when fiber orientation vectors are provided as input
        to the pipeline

    skip_odf: bool
        True when no ODF analysis is required following
        the Frangi filtering stage

    Returns
    -------
    save_subdirs: list (dtype=str)
        saving subdirectory string paths
    """
    # get current time
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # get base path
    base_path = path.dirname(img_path)

    # create saving directory
    save_dir = path.join(base_path, time_stamp + '_' + img_name)
    save_subdirs = []
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # create Frangi filter output subdirectory
    if not skip_frangi:
        frangi_dir = path.join(save_dir, 'frangi')
        mkdir(frangi_dir)
        save_subdirs.append(frangi_dir)
    else:
        save_subdirs.append(None)

    # create ODF analysis output subdirectory
    if not skip_odf:
        odf_dir = path.join(save_dir, 'odf')
        mkdir(odf_dir)
        save_subdirs.append(odf_dir)
    else:
        save_subdirs.append(None)

    return save_subdirs


def save_array(fname, save_dir, nd_array, px_size=None, format='tif', odi=False):
    """
    Save array to file.

    Parameters
    ----------
    fname: string
        output filename (without extension)

    save_dir: string
        saving directory string path

    nd_array: numpy.ndarray
        data

    px_size: tuple
        pixel size (Z,Y,X) [um]

    format: str
        output format

    odi: bool
        True when saving the ODI maps

    Returns
    -------
    None
    """

    # check output format
    format = format.lower()
    if format == 'tif' or format == 'tiff':

        # retrieve image pixel size
        px_size_z, px_size_y, px_size_x = px_size

        # adjust bigtiff optional argument
        bigtiff = True if nd_array.itemsize * np.prod(nd_array.shape) >= 4294967296 else False

        # save array to TIFF file
        if odi:
            tiff.imwrite(path.join(save_dir, fname + '.' + format), nd_array, imagej=True, bigtiff=bigtiff,
                         resolution=(1 / px_size_x, 1 / px_size_y),
                         metadata={'axes': 'ZYX', 'spacing': px_size_z, 'unit': 'um'}, compression='zlib')
        else:
            tiff.imwrite(path.join(save_dir, fname + '.' + format), nd_array, imagej=True, bigtiff=bigtiff,
                         resolution=(1 / px_size_x, 1 / px_size_y),
                         metadata={'spacing': px_size_z, 'unit': 'um'}, compression='zlib')

    # save array to NumPy file
    elif format == 'npy':
        np.save(path.join(save_dir, fname + '.npy'), nd_array)

    # save array to NIfTI file
    elif format == 'nii':
        nd_array = nib.Nifti1Image(nd_array, np.eye(4))
        nd_array.to_filename(path.join(save_dir, fname + '.nii'))

    # raise error
    else:
        raise ValueError("  Unsupported data format!!!")
