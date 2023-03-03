import gc
from multiprocessing import cpu_count
from os import environ, path, remove, unlink
from shutil import rmtree
from time import perf_counter

import numpy as np
from astropy.visualization import make_lupton_rgb
from joblib import dump, load
from matplotlib.colors import hsv_to_rgb
from skimage.filters import (threshold_li, threshold_niblack,
                             threshold_sauvola, threshold_triangle,
                             threshold_yen)
from skimage.morphology import skeletonize_3d


def create_background_mask(img, thresh_method='yen', skeletonize=False):
    """
    Compute background mask.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    thresh_method: str
        image thresholding method

    skeletonize: bool
        if True, apply skeletonization to the boolean mask of myelinated fibers

    Returns
    -------
    background_mask: numpy.ndarray (shape=(Z,Y,X), dtype=bool)
        boolean background mask
    """
    # select thresholding method
    if thresh_method == 'li':
        initial_li_guess = np.mean(img[img != 0])
        thresh = threshold_li(img, initial_guess=initial_li_guess)
    elif thresh_method == 'niblack':
        thresh = threshold_niblack(img, window_size=15, k=0.2)
    elif thresh_method == 'sauvola':
        thresh = threshold_sauvola(img, window_size=15, k=0.2, r=None)
    elif thresh_method == 'triangle':
        thresh = threshold_triangle(img, nbins=256)
    elif thresh_method == 'yen':
        thresh = threshold_yen(img, nbins=256)
    else:
        raise ValueError("  Unsupported thresholding method!!!")

    # compute mask
    background_mask = img < thresh

    # skeletonize mask
    if skeletonize:
        background_mask = np.logical_not(background_mask)
        background_mask = skeletonize_3d(background_mask)
        background_mask = np.logical_not(background_mask)

    return background_mask


def create_memory_map(file_path, shape, dtype, arr=None, mmap_mode='r+'):
    """
    Create a memory-map to an array stored in a binary file on disk.

    Parameters
    ----------
    file_path: str
        path to file object to be used as the array data buffer

    shape: tuple
        shape of the store array

    dtype:
        data-type used to interpret the file contents

    arr: numpy.ndarray
        array to be mapped

    mmap_mode: str
        file opening mode

    Returns
    -------
    mmap: NumPy memory map
        memory-mapped array
    """
    if path.exists(file_path):
        unlink(file_path)
    if arr is None:
        arr = np.zeros(tuple(shape), dtype=dtype)
    _ = dump(arr, file_path)
    mmap = load(file_path, mmap_mode=mmap_mode)
    del arr
    _ = gc.collect()

    return mmap


def get_available_cores():
    """
    Return the number of available logical cores.

    Parameters
    ----------
    None

    Returns
    -------
    num_cpu: int
        number of available cores
    """
    num_cpu = environ.pop('OMP_NUM_THREADS', default=None)
    if num_cpu is None:
        num_cpu = cpu_count()
    else:
        num_cpu = int(num_cpu)

    return num_cpu


def delete_tmp_folder(tmp_dir):
    """
    Delete temporary folder.

    Parameters
    ----------
    tmp_dir: str
        path to temporary folder to be removed

    Returns
    -------
    None
    """
    try:
        rmtree(tmp_dir)
    except OSError:
        pass


def delete_tmp_files(file_lst):
    """
    Close and remove temporary files.

    Parameters
    ----------
    file_lst: list
        list of temporary file dictionaries
        ('path': file path; 'obj': file object)

    Returns
    -------
    None
    """
    if type(file_lst) is not list:
        file_lst = [file_lst]

    for file in file_lst:
        file['obj'].close()
        remove(file['path'])


def divide_nonzero(nd_array1, nd_array2, new_value=1e-10):
    """
    Divide two arrays handling zero denominator values.

    Parameters
    ----------
    nd_array1: numpy.ndarray
        dividend array

    nd_array2: numpy.ndarray
        divisor array

    new_value: float
        substituted value

    Returns
    -------
    divided: numpy.ndarray
        divided array
    """
    denominator = np.copy(nd_array2)
    denominator[denominator == 0] = new_value
    divided = np.divide(nd_array1, denominator)

    return divided


def elapsed_time(start_time):
    """
    Compute elapsed time from input start reference.

    Parameters
    ----------
    start_time: float
        start time reference

    Returns
    -------
    total: float
        total time [s]

    mins: float
        minutes

    secs: float
        seconds
    """
    stop_time = perf_counter()
    total = stop_time - start_time
    mins = total // 60
    secs = total % 60

    return total, mins, secs


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


def get_item_bytes(data):
    """
    Retrieve data item size in bytes.

    Parameters
    ----------
    data: numpy.ndarray or HDF5 dataset
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


def get_output_prefix(scales_um, alpha, beta, gamma):
    """
    Generate the output filename prefix including
    pipeline configuration information.

    Parameters
    ----------
    scales_um: list (dtype=float)
        analyzed spatial scales in [μm]

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    Returns
    -------
    pfx: str
        pipeline configuration prefix
    """
    # generate prefix
    pfx = 'sc'
    for s in scales_um:
        pfx = pfx + str(s) + '_'
    pfx = 'a' + str(alpha) + '_b' + str(beta) + '_g' + str(gamma) + '_' + pfx

    return pfx


def normalize_angle(angle, lower=0.0, upper=360.0, dtype=None):
    """
    Normalize angle to [lower, upper) range.

    Parameters
    ----------
    angle: float or list (dtype=float)
        angular value(s) to be normalized (in degrees)

    lower: float
        lower limit (default: 0.0)

    upper: float
        upper limit (default: 360.0)

    dtype:
        output data type

    Returns
    -------
    norm_angle: float or list (dtype=float)
        angular value(s) (in degrees) normalized within [lower, upper)

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
        raise ValueError("  Invalid lower and upper limits: (%s, %s)" % (lower, upper))

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


def normalize_image(img, max_out_val=255.0, dtype=np.uint8):
    """
    Normalize image data.

    Parameters
    ----------
    img: numpy.ndarray
        input image

    max_out_val: float
        maximum output value

    dtype:
        output data type

    Returns
    -------
    norm_img: numpy.ndarray
        normalized image
    """
    # get min and max values
    min_val = np.min(img)
    max_val = np.max(img)

    # normalization
    if max_val != 0:
        if max_val != min_val:
            norm_img = (((img - min_val) / (max_val - min_val)) * max_out_val).astype(dtype)
        else:
            norm_img = ((img / max_val) * max_out_val).astype(dtype)
    else:
        norm_img = img.astype(dtype)

    return norm_img


def orient_colormap(vec_img):
    """
    Compute HSV colormap of vector orientations from 3D vector field.

    Parameters
    ----------
    vec_img: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        orientation vectors

    Returns
    -------
    rgb_map: numpy.ndarray (shape=(Z,Y,X,3), dtype=uint8)
        orientation color map
    """
    # get input array shape
    vec_img_shape = vec_img.shape

    # select planar components
    vy = vec_img[..., 1]
    vx = vec_img[..., 2]

    # compute the in-plane versor length
    vxy_abs = np.sqrt(np.square(vx) + np.square(vy))
    vxy_abs = np.divide(vxy_abs, np.max(vxy_abs))

    # compute the in-plane angular orientation
    vxy_ang = normalize_angle(np.arctan2(vy, vx), lower=0, upper=np.pi, dtype=np.float32)
    vxy_ang = np.divide(vxy_ang, np.pi)

    # initialize colormap
    rgb_map = np.zeros(shape=tuple(list(vec_img_shape[:-1]) + [3]), dtype=np.uint8)
    for z in range(vec_img_shape[0]):

        # generate colormap slice by slice
        h = vxy_ang[z]
        s = vxy_abs[z]
        v = s
        hsv_map = np.stack((h, s, v), axis=-1)

        # conversion to 8-bit precision
        rgb_map[z] = (255.0 * hsv_to_rgb(hsv_map)).astype(np.uint8)

    return rgb_map


def ceil_to_multiple(number, multiple):
    """
    Round up number to the nearest multiple.

    Parameters
    ----------
    number:
        number to be rounded

    multiple:
        the input number will be rounded
        to the nearest multiple higher than this value

    Returns
    -------
    rounded:
        rounded up number
    """
    rounded = multiple * np.ceil(number / multiple)

    return rounded


def transform_axes(nd_array, flipped=None, swapped=None, expand=None):
    """
    Manipulate axes and dimensions of the input array.
    The transformation sequence is:
    axes flip >>> axes swap >>> dimensions expansion.

    Parameters
    ----------
    nd_array: numpy.ndarray
        input data array

    swapped: tuple (dtype=int)
        axes to be swapped

    flipped: tuple (dtype=int)
        axes to be flipped

    expand: int
        insert new axis at this position

    Returns
    -------
    nd_array: numpy.ndarray
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


def vector_colormap(vec_img):
    """
    Compute RGB colormap of orientation vector components from 3D vector field.

    Parameters
    ----------
    vec_img: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        n-dimensional array of orientation vectors

    Returns
    -------
    rgb_map: numpy.ndarray (shape=(Z,Y,X,3), dtype=uint8)
        orientation color map
    """
    # get input array shape
    vec_img_shape = vec_img.shape

    # take absolute value
    vec_img = np.abs(vec_img)

    # initialize colormap
    rgb_map = np.zeros(shape=vec_img_shape, dtype=np.uint8)
    for z in range(vec_img_shape[0]):

        # generate colormap slice by slice
        img_r = vec_img[z, :, :, 2]
        img_g = vec_img[z, :, :, 1]
        img_b = vec_img[z, :, :, 0]
        rgb_map[z] = make_lupton_rgb(img_r, img_g, img_b, minimum=0, stretch=1, Q=8)

    return rgb_map
