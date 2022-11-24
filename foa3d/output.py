from datetime import datetime
from os import mkdir, path

import nibabel as nib
import numpy as np
import tifffile as tiff


def create_save_dir(image_path):
    """
    Create saving directory.

    Parameters
    ----------
    image_path: str
        path to input microscopy volume image

    Returns
    -------
    save_dir: str
        saving directory string path
    """
    # get current time
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # get volume image name
    base_path = path.dirname(image_path)
    image_fullname = path.basename(image_path)
    image_name = image_fullname.split('.')[0]

    # create saving directory
    save_dir = path.join(base_path, time_stamp + '_' + image_name)
    if not path.isdir(save_dir):
        mkdir(save_dir)

    return save_dir


def save_array(fname, save_dir, nd_array, format='tif', odi=False):
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

    format: str
        output format

    odi: bool
        True when saving the ODI maps

    Returns
    -------
    None
    """
    format = format.lower()
    if format == 'tif' or format == 'tiff':
        if odi:
            tiff.imwrite(path.join(save_dir, fname + '.' + format), nd_array, imagej=True, metadata={'axes': 'ZYX'})
        else:
            tiff.imwrite(path.join(save_dir, fname + '.' + format), nd_array)
    elif format == 'npy':
        np.save(path.join(save_dir, fname + '.npy'), nd_array)
    elif format == 'nii':
        nd_array = nib.Nifti1Image(nd_array, np.eye(4))
        nd_array.to_filename(path.join(save_dir, fname + '.nii'))
    else:
        raise ValueError("  Unsupported data format!!!")
