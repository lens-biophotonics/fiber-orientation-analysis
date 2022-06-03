import gc
from os import remove

import numpy as np
from astropy.visualization import make_lupton_rgb
from h5py import File
from matplotlib.colors import hsv_to_rgb
from skimage.filters import (threshold_li, threshold_niblack,
                             threshold_sauvola, threshold_triangle,
                             threshold_yen)
from skimage.morphology import skeletonize_3d


def create_background_mask(volume, thresh_method='yen', skeletonize=False):
    """
    Compute background mask.

    Parameters
    ----------
    volume: ndarray (shape: (Z,Y,X))
        input TPFM image volume

    thresh_method: string
        image thresholding method

    skeletonize: bool
        if True, thin the resulting binary mask

    Returns
    -------
    background_mask: ndarray (dtype: bool)
        boolean background mask
    """
    # select thresholding method
    if thresh_method == 'li':
        initial_li_guess = np.mean(volume[volume != 0])
        thresh = threshold_li(volume, initial_guess=initial_li_guess)
    elif thresh_method == 'niblack':
        thresh = threshold_niblack(volume, window_size=15, k=0.2)
    elif thresh_method == 'sauvola':
        thresh = threshold_sauvola(volume, window_size=15, k=0.2, r=None)
    elif thresh_method == 'triangle':
        thresh = threshold_triangle(volume, nbins=256)
    elif thresh_method == 'yen':
        thresh = threshold_yen(volume, nbins=256)
    else:
        raise ValueError("  Unsupported thresholding method!!!")

    # compute mask
    background_mask = volume < thresh

    # skeletonize mask
    if skeletonize is True:
        background_mask = np.logical_not(background_mask)
        background_mask = skeletonize_3d(background_mask)
        background_mask = np.logical_not(background_mask)

    return background_mask


def vector_colormap(vec_volume):
    """
    Compute RGB colormap of orientation vector components from
    the input 3D vector field.

    Parameters
    ----------
    vec_volume: ndarray (shape=(Z,Y,X,3), dtype: float)
        n-dimensional array of orientation vectors

    Returns
    -------
    rgb_map: ndarray (shape=(Z,Y,X,3), dtype: uint8)
        orientation color map
    """
    # get eigenvectors array shape
    vec_volume_shape = vec_volume.shape

    # take absolute value
    vec_volume = np.abs(vec_volume)

    # initialize colormap
    rgb_map = np.zeros(shape=vec_volume_shape, dtype=np.uint8)
    for z in range(vec_volume_shape[0]):

        # generate colormap slice by slice
        image_r = vec_volume[z, :, :, 2]
        image_g = vec_volume[z, :, :, 1]
        image_b = vec_volume[z, :, :, 0]
        rgb_map[z] = make_lupton_rgb(image_r, image_g, image_b, minimum=0,
                                     stretch=1, Q=8)

    return rgb_map


def orient_colormap(vec_volume):
    """
    Compute HSV colormap of vector orientations from the input 3D vector field.

    Parameters
    ----------
    vec_volume: ndarray (shape=(Z,Y,X,3), dtype: float)
        n-dimensional array of orientation vectors

    Returns
    -------
    rgb_map: ndarray (shape=(Z,Y,X,3), dtype: uint8)
        orientation color map
    """
    # get eigenvectors array shape
    vec_volume_shape = vec_volume.shape

    # select planar components
    vy = vec_volume[..., 1]
    vx = vec_volume[..., 2]

    # compute the in-plane versor length
    vxy_abs = np.sqrt(np.square(vx) + np.square(vy))
    vxy_abs = np.divide(vxy_abs, np.max(vxy_abs))

    # compute the in-plane angular orientation
    vxy_ang = normalize_angle(np.arctan2(vy, vx), lower=0, upper=np.pi,
                              dtype=np.float32)
    vxy_ang = np.divide(vxy_ang, np.pi)

    # initialize colormap
    rgb_map = np.zeros(shape=tuple(list(vec_volume_shape[:-1]) + [3]),
                       dtype=np.uint8)
    for z in range(vec_volume_shape[0]):

        # generate colormap slice by slice
        h = vxy_ang[z]
        s = vxy_abs[z]
        v = s
        hsv_map = np.stack((h, s, v), axis=-1)

        # conversion to 8-bit precision
        rgb_map[z] = (255.0 * hsv_to_rgb(hsv_map)).astype(np.uint8)

    return rgb_map


def normalize_angle(angle, lower=0.0, upper=360.0, dtype=None):
    """
    Normalize angle to [lower, upper) range.

    Parameters
    ----------
    angle: float
        value to be normalized (in degrees)

    lower: float
        lower limit (default: 0.0)

    upper: float
        upper limit (default: 360.0)

    dtype:
        output data type

    Returns
    -------
    norm_angle: float
        angle normalized within [lower, upper)

    Raises
    ------
    ValueError
      if lower >= upper
    """
    # convert to array if needed
    isList = False
    if np.isscalar(angle):
        angle = np.array(angle)
    elif isinstance(angle, list):
        angle = np.array(angle)
        isList = True

    # check limits
    if lower >= upper:
        raise ValueError("Invalid lower and upper limits: (%s, %s)" %
                         (lower, upper))

    # view
    norm_angle = angle

    # correction 1
    c1 = np.logical_or(angle > upper, angle == lower)
    angle[c1] = lower + abs(angle[c1] + upper) % (abs(lower) + abs(upper))

    # correction 2
    c2 = np.logical_or(angle < lower, angle == upper)
    angle[c2] = upper - abs(angle[c2] - lower) % (abs(lower) + abs(upper))

    # correction 3
    angle[angle == upper] = lower

    # cast to desired data type
    if dtype is not None:
        norm_angle = norm_angle.astype(dtype)

    # convert back to list
    if isList:
        norm_angle = list(norm_angle)

    return norm_angle


def normalize_image(image, max_out_value=255.0, dtype=np.uint8):
    """
    Normalize image data.

    Parameters
    ----------
    image: ndarray
        input image

    max_out_value: float
        maximum output value

    dtype:
        output data type

    Returns
    -------
    norm_image: ndarray
        normalized image
    """
    # get min and max values
    min_value = np.min(image)
    max_value = np.max(image)

    # normalization
    if max_value != 0:
        if max_value != min_value:
            norm_image = (((image - min_value) / (max_value - min_value)) *
                          max_out_value).astype(dtype)
        else:
            norm_image = ((image / max_value) * max_out_value).astype(dtype)
    else:
        norm_image = image.astype(dtype)

    return norm_image


def transform_axes(nd_array, flipped=None, swapped=None, expand=None):
    """
    Manipulate axes and dimensions of input data array.
    The transformation sequence is:
    axes flip >>> axes swap >>> dimensions expansion.

    Parameters
    ----------
    nd_array: ndarray
        input data array

    swapped: tuple (dtype=int)
        axes to be swapped

    flipped: tuple (dtype=int)
        axes to be flipped

    expand: int
        insert new axis at this position

    Returns
    -------
    nd_array: ndarray
        transformed data array
    """
    if flipped is not None:
        nd_array = np.flip(nd_array, axis=flipped)

    if swapped is not None:
        swap_src, swap_dest = swapped
        nd_array = np.swapaxes(nd_array, swap_src, swap_dest)

    if expand is not None:
        nd_array = np.expand_dims(nd_array, axis=expand)

    return nd_array


def clear_from_memory(data):
    del data
    gc.collect()


def create_hdf5_file(path, dset_shape, chunk_shape, dtype):
    """
    Create HDF5 dataset.

    Parameters
    ----------
    path: path object
        HDF5 file path

    dset_shape: tuple (dtype: int)
        overall dataset shape

    chunk_shape: tuple (dtype: int)
        shape of the chunked storage layout

    dtype:
        data type of the HDF5 dataset

    Returns
    -------
    file: HDF5 file object

    dset: HDF5 dataset
    """
    file = File(path, 'w')
    dset = file.create_dataset('chunked', tuple(dset_shape),
                               chunks=tuple(chunk_shape), dtype=dtype)

    return file, dset


def delete_tmp_files(file_list):
    """
    Close and remove temporary files.

    Parameters
    ----------
    file_list: list
        list of temporary file dictionaries
        ('path': file path; 'obj': file object)

    Returns
    -------
    None
    """
    if type(file_list) is not list:
        file_list = [file_list]

    for file in file_list:
        file['obj'].close()
        remove(file['path'])


def get_item_bytes(data):
    """
    Retrieve data item size in bytes.

    Parameters
    ----------
    data: ndarray or HDF5 dataset
        input data

    Returns
    -------
    bytes: int
        item size in bytes
    """
    # get data type
    data_type = data.dtype

    # type byte size
    try:
        bytes = int(np.iinfo(data_type).bits / 8)
    except ValueError:
        bytes = int(np.finfo(data_type).bits / 8)

    return bytes


def divide_nonzero(nd_array1, nd_array2, new_value=1e-10):
    """
    Divide two arrays handling zero denominator values.

    Parameters
    ----------
    nd_array1: ndarray
        dividend array

    nd_array2: ndarray
        divisor array

    new_value: float
        substituted value

    Returns
    -------
    divided: ndarray
        divided array
    """
    denominator = np.copy(nd_array2)
    denominator[denominator == 0] = new_value
    divided = np.divide(nd_array1, denominator)

    return divided


def round_to_multiple(number, multiple):
    """
    Round number to the nearest multiple.
    """
    return multiple * np.round(number / multiple)


def fwhm_to_sigma(fwhm):
    """
    Compute the standard deviation of a Gaussian distribution
    from its FWHM value.

    Parameters
    ----------
    fwhm: float
        full width at half maximum

    Returns
    -------
    sigma: float
        standard deviation
    """
    sigma = np.sqrt(np.square(fwhm) / (8 * np.log(2)))

    return sigma
