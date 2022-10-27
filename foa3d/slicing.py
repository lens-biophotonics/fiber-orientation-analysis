import numpy as np
from foa3d.utils import round_to_multiple


def adjust_slice_coord(axis_iter, pad_rng, slice_shape, image_shape, axis, flip=False):
    """
    Adjust image slice coordinates at boundaries.

    Parameters
    ----------
    axis_iter: int
        iteration counter along axis (see pipeline.iterate_frangi_on_slices)

    pad_rng: int
        patch padding range

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    image_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    axis: int
        axis index

    flip: bool
        if True, flip axes coordinates

    Returns
    -------
    start: int
        adjusted start index

    stop: int
        adjusted stop index

    axis_pad_array: numpy.ndarray (shape=(2,), dtype=int)
        axis pad range [\'left\', \'right\']
    """
    # initialize axis pad array
    axis_pad_array = np.zeros(shape=(2,), dtype=np.uint8)

    # compute start and stop coordinates
    start = axis_iter * slice_shape[axis] - pad_rng
    stop = axis_iter * slice_shape[axis] + slice_shape[axis] + pad_rng

    # flip coordinates
    if flip:
        start_tmp = start
        start = image_shape[axis] - stop
        stop = image_shape[axis] - start_tmp

    # adjust start coordinate
    if start < 0:
        start = 0
    else:
        axis_pad_array[0] = pad_rng

    # adjust stop coordinate
    if stop > image_shape[axis]:
        stop = image_shape[axis]
    else:
        axis_pad_array[1] = pad_rng

    return start, stop, axis_pad_array


def compute_slice_range(z, y, x, slice_shape, image_shape, pad_rng=0, flip=False):
    """
    Compute basic slice coordinates from microscopy volume image.

    Parameters
    ----------
    z: int
        z-depth index

    y: int
        row index

    x: int
        column index

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    image_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    pad_rng: int
        slice padding range

    flip: bool
        if True, flip axes coordinates

    Returns
    -------
    rng: tuple
        3D slice index ranges

    pad_mat: numpy.ndarray
        3D padding range array
    """
    # adjust original image patch coordinates
    # and generate padding range matrix
    pad_mat = np.zeros(shape=(3, 2), dtype=np.uint8)
    z_start, z_stop, pad_mat[0, :] = adjust_slice_coord(z, pad_rng, slice_shape, image_shape, axis=0, flip=flip)
    y_start, y_stop, pad_mat[1, :] = adjust_slice_coord(y, pad_rng, slice_shape, image_shape, axis=1, flip=flip)
    x_start, x_stop, pad_mat[2, :] = adjust_slice_coord(x, pad_rng, slice_shape, image_shape, axis=2, flip=flip)

    # generate index ranges
    z_rng = slice(z_start, z_stop, 1)
    y_rng = slice(y_start, y_stop, 1)
    x_rng = slice(x_start, x_stop, 1)
    rng = np.index_exp[z_rng, y_rng, x_rng]

    return rng, pad_mat


def compute_slice_shape(image_shape, image_item_size, px_size=None, max_slice_size=100):
    """
    Compute basic image chunk shape depending on its maximum size (in bytes).

    Parameters
    ----------
    image_shape: numpy.ndarray (shape=(Z,Y,X) or shape=(Z,Y,X,3))
        total image shape [px]

    image_item_size: int
        image item size (in bytes)

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    max_slice_size: float
        maximum memory size (in bytes) of the basic slices analyzed iteratively

    Returns
    -------
    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the basic image slices analyzed iteratively [μm]
        (if px_size is provided)
    """
    slice_depth = image_shape[0]
    slice_side = np.round(1024 * np.sqrt((max_slice_size / (slice_depth * image_item_size))))
    slice_shape = np.array([slice_depth, slice_side, slice_side]).astype(int)
    slice_shape = np.min(np.stack((image_shape[:3], slice_shape)), axis=0)

    if px_size is not None:
        slice_shape_um = np.multiply(slice_shape, px_size)
        return slice_shape, slice_shape_um
    else:
        return slice_shape


def compute_smoothing_pad_range(smooth_sigma, truncate=4):
    """
    Compute lateral image padding range
    for coping with smoothing-related boundary artifacts.

    Parameters
    ----------
    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (applied to the XY plane)

    truncate: int
        truncate the Gaussian kernel at this many standard deviations

    Returns
    -------
    pad_rng: int
        slice padding range
    """
    if smooth_sigma is not None:
        kernel_size = (np.ceil(2 * truncate * smooth_sigma) // 2 * 2 + 1).astype(int)
        pad_rng = np.max(kernel_size)
    else:
        pad_rng = 0

    return pad_rng


def config_frangi_slicing(image_shape, image_item_size, px_size, px_size_iso, smooth_sigma, max_slice_size=100.0):
    """
    Slicing configuration for the iterative Frangi filtering of basic chunks
    of the input microscopy volume.

    Parameters
    ----------
    image_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    image_item_size: int
        image item size (in bytes)

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    max_slice_size: float
        maximum memory size (in bytes) of the basic image slices
        analyzed iteratively

    Returns
    -------
    in_slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    in_slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the basic image slices analyzed iteratively [μm]

    out_slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the processed image slices [px]

    out_image_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the whole output image [px]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    pad: int
        padding range
    """
    # shape of processed TPFM slices
    in_slice_shape, in_slice_shape_um = \
        compute_slice_shape(image_shape, image_item_size, px_size=px_size, max_slice_size=max_slice_size)

    # adjust shapes according to the anisotropy correction
    px_rsz_ratio = np.divide(px_size, px_size_iso)
    out_slice_shape = np.ceil(px_rsz_ratio * in_slice_shape).astype(int)
    out_image_shape = np.ceil(px_rsz_ratio * image_shape).astype(int)

    # compute input patch padding range (border artifacts suppression)
    pad = compute_smoothing_pad_range(smooth_sigma)

    return in_slice_shape, in_slice_shape_um, out_slice_shape, out_image_shape, px_rsz_ratio, pad


def config_odf_slicing(fiber_vec_shape, fiber_vec_item_size, px_size_iso, odf_scale_um, max_slice_size=100.0):
    """
    Slicing configuration for the iterative ODF analysis of basic chunks
    of the 3D fiber orientation map returned by the Frangi filter stage.

    Parameters
    ----------
    fiber_vec_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the fiber vector dataset (component axis excluded) [px]

    fiber_vec_item_size: int
        vector dataset item size (in bytes)

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    max_slice_size: float
        maximum memory size (in bytes) of the basic image slices
        analyzed iteratively

    Returns
    -------
    fiber_vec_slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the analyzed fiber vector slices [px]

    odf_slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the resulting ODF map slices [px]

    odf_scale: int
        fiber ODF resolution (super-voxel side in [px])
    """
    # shape of processed TPFM slices
    fiber_vec_slice_shape, _ = \
        compute_slice_shape(fiber_vec_shape, fiber_vec_item_size, px_size=px_size_iso, max_slice_size=max_slice_size)

    # derive the ODF map shape from the ODF kernel size
    odf_scale = int(np.ceil(odf_scale_um / px_size_iso[0]))

    # adapt lateral vector chunk shape (Z,Y,X)
    fiber_vec_slice_shape[1] = round_to_multiple(fiber_vec_slice_shape[1], odf_scale)
    fiber_vec_slice_shape[2] = fiber_vec_slice_shape[1]

    # get output ODF chunk shape (X,Y,Z)
    odf_slice_shape = np.ceil(np.divide(fiber_vec_slice_shape, odf_scale)).astype(int)

    return fiber_vec_slice_shape, odf_slice_shape, odf_scale


def crop_slice(img_slice, rng, flipped=()):
    """
    Shrink image slice at volume boundaries, for overall shape consistency.

    Parameters
    ----------
    img_slice: numpy.ndarray
        image slice

    rng: tuple
        3D index range

    flipped: tuple
        flipped axes

    Returns
    -------
    cropped_slice: numpy.ndarray
        cropped image slice
    """
    # check slice shape and output index ranges
    out_slice_shape = img_slice.shape
    crop_rng = np.zeros(shape=(3,), dtype=int)
    for s in range(3):
        crop_rng[s] = out_slice_shape[s] - np.arange(rng[s].start, rng[s].stop, rng[s].step).size

    # crop slice if required
    if 0 in flipped:
        cropped_slice = img_slice[crop_rng[0] or None:, ...]
    else:
        cropped_slice = img_slice[:-crop_rng[0] or None, ...]
    if 1 in flipped:
        cropped_slice = cropped_slice[:, crop_rng[1] or None:, ...]
    else:
        cropped_slice = cropped_slice[:, :-crop_rng[1] or None, ...]
    if 2 in flipped:
        cropped_slice = cropped_slice[:, :, crop_rng[2] or None:, ...]
    else:
        cropped_slice = cropped_slice[:, :, :-crop_rng[2] or None, ...]

    return cropped_slice


def slice_channel(image, rng, channel, mosaic=False):
    """
    Slice desired channel from input image volume.

    Parameters
    ----------
    image: numpy.ndarray
        microscopy volume image

    rng: tuple (dtype=int)
        3D index ranges

    channel: int
        image channel axis

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    image_slice: numpy.ndarray
        sliced image patch
    """
    z_rng, r_rng, c_rng = rng

    if mosaic:
        image_slice = image[z_rng, channel, r_rng, c_rng]
    else:
        image_slice = image[z_rng, r_rng, c_rng, channel]

    return image_slice
