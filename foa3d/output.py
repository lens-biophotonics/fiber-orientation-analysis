import psutil
import tempfile

from datetime import datetime
from os import makedirs, path

import nibabel as nib
import numpy as np
from tifffile import TiffWriter

from foa3d.utils import get_item_size


def create_save_dirs(img_path, img_name, cli_args, is_fovec=False):
    """
    Create saving directory.

    Parameters
    ----------
    img_path: str
        path to input microscopy volume image

    img_name: str
        name of the input volume image

    cli_args: see ArgumentParser.parse_args
        updated namespace of command line arguments

    is_fovec: bool
        True when fiber orientation vectors are provided as input
        to the pipeline

    Returns
    -------
    save_subdirs: list (dtype=str)
        saving subdirectory string paths

    tmp_dir: str
        temporary directory (for memory-map objects)
    """

    # get current time
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # get output path
    out_path = cli_args.out
    if out_path is None:
        out_path = path.dirname(img_path)

    # create saving directory
    base_out_dir = path.join(out_path, f"Foa3D_{time_stamp}_{img_name}")
    save_dir_lst = list()
    if not path.isdir(base_out_dir):
        makedirs(base_out_dir)

    # create Frangi filter output subdirectory
    if not is_fovec:
        frangi_dir = path.join(base_out_dir, 'frangi')
        makedirs(frangi_dir)
        save_dir_lst.append(frangi_dir)
    else:
        save_dir_lst.append(None)

    # create ODF analysis output subdirectory
    if cli_args.odf_res is not None:
        odf_dir = path.join(base_out_dir, 'odf')
        makedirs(odf_dir)
        save_dir_lst.append(odf_dir)
    else:
        save_dir_lst.append(None)

    # create temporary directory
    tmp_dir = tempfile.mkdtemp(dir=out_path)

    return save_dir_lst, tmp_dir


def save_array(fname, save_dir, nd_array, px_sz=None, fmt='tiff', ram=None):
    """
    Save array to file.

    Parameters
    ----------
    fname: string
        output filename

    save_dir: string
        saving directory string path

    nd_array: NumPy memory-map object or HDF5 dataset
        data

    px_sz: tuple
        pixel size (Z,Y,X) [um]

    fmt: str
        output format

    ram: float
        maximum RAM available

    Returns
    -------
    None
    """

    # get maximum RAM and initialized array memory size
    if ram is None:
        ram = psutil.virtual_memory()[1]
    itm_sz = get_item_size(nd_array.dtype)
    dz = np.floor(ram / (itm_sz * np.prod(nd_array.shape[1:]))).astype(int)
    nz = np.ceil(nd_array.shape[0] / dz).astype(int)

    # check output format
    fmt = fmt.lower()
    if fmt == 'tif' or fmt == 'tiff':

        # retrieve image pixel size
        px_sz_z, px_sz_y, px_sz_x = px_sz

        # adjust bigtiff optional argument
        bigtiff = True if nd_array.itemsize * nd_array.size >= 4294967296 else False
        out_name = f'{fname}.{fmt}'
        with TiffWriter(path.join(save_dir, out_name), bigtiff=bigtiff, append=True) as tif:
            for z in range(nz):
                zs = z * dz
                tif.write(nd_array[zs:zs + dz, ...], contiguous=True, resolution=(1 / px_sz_x, 1 / px_sz_y),
                          metadata={'spacing': px_sz_z, 'unit': 'um'})

    # save array to NumPy file
    elif fmt == 'npy':
        np.save(path.join(save_dir, fname + '.npy'), nd_array)

    # save array to NIfTI file
    elif fmt == 'nii':
        nd_array = nib.Nifti1Image(nd_array, np.eye(4))
        nd_array.to_filename(path.join(save_dir, fname + '.nii'))

    # raise error
    else:
        raise ValueError("Unsupported data format!!!")
