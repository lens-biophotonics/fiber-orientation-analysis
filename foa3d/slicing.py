import numpy as np
from foa3d.utils import round_to_multiple


def config_frangi_slicing(volume_shape, volume_item_size, px_size, px_size_iso, smooth_sigma, max_slice_size=100.0):
    """
    Slicing configuration for the iterative Frangi filtering of basic chunks
    of the input microscopy volume.

    Parameters
    ----------
    volume_shape: ndarray (shape=(3,), dtype=int)
        volume shape [px]

    volume_item_size: int
        array item size (in bytes)

    px_size: ndarray (shape=(3,), dtype=float)
        original TPFM pixel size [μm]

    px_size_iso: ndarray (shape=(3,), dtype=float)
        isotropic TPFM pixel size [μm]

    smooth_sigma: ndarray (shape=(3,), dtype=int)
        3D standard deviation of low-pass Gaussian filter [px]

    max_slice_size: float
        maximum memory size (in bytes) of the basic slices analyzed iteratively

    Returns
    -------
    in_chunk_shape: ndarray (shape=(3,), dtype=int)
        shape of the input image patches [px]

    in_chunk_shape_um: ndarray (shape=(3,), dtype=float)
        shape of the input image patches [μm]

    out_chunk_shape: ndarray (shape=(3,), dtype=int)
        shape of the processed image patches [px]

    resize_ratio: ndarray (shape=(3,), dtype=float)
        3D axes resize ratio

    pad: int
        patch padding range
    """
    # shape of processed TPFM slices
    in_chunk_shape, in_chunk_shape_um = \
        compute_chunk_shape(volume_shape, volume_item_size, px_size=px_size, max_slice_size=max_slice_size)

    # adjust shapes according to the anisotropy correction
    px_rsz_ratio = np.divide(px_size, px_size_iso)
    out_chunk_shape = np.ceil(px_rsz_ratio * in_chunk_shape).astype(int)
    out_volume_shape = np.ceil(px_rsz_ratio * volume_shape).astype(int)

    # compute input patch padding range (border artifacts suppression)
    pad = compute_smoothing_pad_range(smooth_sigma)

    return in_chunk_shape, in_chunk_shape_um, out_chunk_shape, out_volume_shape, px_rsz_ratio, pad


def config_odf_slicing(vec_volume_shape, vec_item_size, px_size_iso, odf_scale_um, max_slice_size=100.0):
    """
    Description.

    Parameters
    ----------
    vec_volume_shape: ndarray (shape=(3,), dtype=int)
        vector dataset shape [px] (vector components axis excluded)

    vec_item_size: int
        vector dataset item size (in bytes)

    px_size_iso: ndarray (shape=(3,), dtype=float)
        isotropic TPFM pixel size [μm]

    odf_scale_um: float
        fiber ODF resolution (super-voxel side in [μm])

    max_slice_size: float
        maximum memory size (in bytes) of the basic slices analyzed iteratively

    Returns
    -------
    vec_patch_shape: ndarray (shape=(3,), dtype=int)
        shape of the analyzed vector patches [px]

    odf_patch_shape: ndarray (shape=(3,), dtype=int)
        shape of the resulting ODF map patches [px]

    odf_scale: int
        fiber ODF resolution (super-voxel side in [px])
    """
    # shape of processed TPFM slices
    vec_patch_shape, _ = \
        compute_chunk_shape(vec_volume_shape, vec_item_size, px_size=px_size_iso, max_slice_size=max_slice_size)

    # derive the ODF map shape from the ODF kernel size
    odf_scale = int(np.ceil(odf_scale_um / px_size_iso[0]))

    # adapt lateral vector chunk shape (Z,Y,X)
    vec_patch_shape[1] = round_to_multiple(vec_patch_shape[1], odf_scale)
    vec_patch_shape[2] = vec_patch_shape[1]

    # get output ODF chunk shape (X,Y,Z)
    odf_patch_shape = np.ceil(np.divide(vec_patch_shape, odf_scale)).astype(int)

    return vec_patch_shape, odf_patch_shape, odf_scale


def compute_chunk_range(z, y, x, chunk_shape, volume_shape, pad_rng=0, flip=False):
    """
    Compute basic patch coordinates from microscopy image volume.

    Parameters
    ----------
    z: int
        z-depth index

    y: int
        row index

    x: int
        column index

    chunk_shape: ndarray (shape=(3,), dtype=int)
        shape of analyzed image patches [px]

    volume_shape: ndarray (shape=(3,), dtype=int)
        total volume shape [px]

    pad_rng: int
        patch padding range

    flip: bool
        if True, flip axes coordinates

    Returns
    -------
    rng: tuple
        chunk 3D index ranges

    pad_mat: ndarray
        3D padding range array
    """
    # adjust original image patch coordinates
    # and generate padding range matrix
    pad_mat = np.zeros(shape=(3, 2), dtype=np.uint8)
    z_start, z_stop, pad_mat[0, :] = adjust_chunk_coord(z, pad_rng, chunk_shape, volume_shape, axis=0, flip=flip)
    y_start, y_stop, pad_mat[1, :] = adjust_chunk_coord(y, pad_rng, chunk_shape, volume_shape, axis=1, flip=flip)
    x_start, x_stop, pad_mat[2, :] = adjust_chunk_coord(x, pad_rng, chunk_shape, volume_shape, axis=2, flip=flip)

    # generate index ranges
    z_rng = slice(z_start, z_stop, 1)
    y_rng = slice(y_start, y_stop, 1)
    x_rng = slice(x_start, x_stop, 1)
    rng = np.index_exp[z_rng, y_rng, x_rng]

    # return padding matrix if required
    return rng, pad_mat


def adjust_chunk_coord(axis_iter, pad_rng, chunk_shape, volume_shape, axis, flip=False):
    """
    Adjust sliced image patch coordinates at volume boundaries.

    Parameters
    ----------
    axis_iter: int
        iteration counter along axis (see pipeline.iterate_frangi_on_slices)

    pad_rng: int
        patch padding range

    chunk_shape: ndarray (shape=(3,), dtype=int)
        shape of analyzed image patches [px]

    volume_shape: ndarray (shape=(3,), dtype=int)
        total volume shape [px]

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

    axis_pad_array: ndarray
        axis pad range [\'left\', \'right\']
    """
    # initialize axis pad array
    axis_pad_array = np.zeros(shape=(2,), dtype=np.uint8)

    # compute start and stop coordinates
    start = axis_iter * chunk_shape[axis] - pad_rng
    stop = axis_iter * chunk_shape[axis] + chunk_shape[axis] + pad_rng

    # flip coordinates
    if flip:
        start_tmp = start
        start = volume_shape[axis] - stop
        stop = volume_shape[axis] - start_tmp

    # adjust start coordinate
    if start < 0:
        start = 0
    else:
        axis_pad_array[0] = pad_rng

    # adjust stop coordinate
    if stop > volume_shape[axis]:
        stop = volume_shape[axis]
    else:
        axis_pad_array[1] = pad_rng

    return start, stop, axis_pad_array


def compute_chunk_shape(volume_shape, volume_item_size, px_size=None, max_slice_size=100):
    """
    Compute basic image chunk shape depending on its maximum size (in bytes).

    Parameters
    ----------
    volume_shape: ndarray (shape=(Z,Y,X) or shape=(Z,Y,X,3))
        total volume shape [px]

    volume_item_size: int
        array item size (in bytes)

    px_size: ndarray (shape=(3,), dtype=float)
        volume pixel size [μm]

    max_slice_size: float
        maximum memory size (in bytes) of the basic slices analyzed iteratively

    Returns
    -------
    chunk_shape: ndarray (shape=(3,), dtype=int)
        shape of analyzed image patches [px]

    chunk_shape_um: ndarray (shape=(3,), dtype=float)
        shape of analyzed image patches [μm]
        (if px_size is provided)
    """
    chunk_depth = volume_shape[0]
    chunk_side = np.round(1024 * np.sqrt((max_slice_size / (chunk_depth * volume_item_size))))
    chunk_shape = np.array([chunk_depth, chunk_side, chunk_side]).astype(int)
    chunk_shape = np.min(np.stack((volume_shape[:3], chunk_shape)), axis=0)

    if px_size is not None:
        chunk_shape_um = np.multiply(chunk_shape, px_size)
        return chunk_shape, chunk_shape_um
    else:
        return chunk_shape


def compute_smoothing_pad_range(smooth_sigma, truncate=4):
    """
    Compute lateral image padding range
    for coping with smoothing-related boundary artifacts.

    Parameters
    ----------
    smooth_sigma: ndarray (shape=(3,))
        3D standard deviation of low-pass Gaussian filter [px]

    truncate: int
        truncate the Gaussian kernel at this many standard deviations
        (default: 4)

    Returns
    -------
    pad_rng: int
        patch padding range
    """
    if smooth_sigma is not None:
        kernel_size = (np.ceil(2 * truncate * smooth_sigma) // 2 * 2 + 1).astype(int)
        pad_rng = np.max(kernel_size)
    else:
        pad_rng = 0

    return pad_rng


def slice_channel(volume, rng, channel, mosaic=False):
    """
    Slice desired channel from input image volume.

    Parameters
    ----------
    volume: ndarray
        input TPFM image volume

    rng: tuple
        3D index ranges

    channel: int
        image channel axis

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    chunk: ndarray
        sliced image patch
    """
    z_rng, r_rng, c_rng = rng
    if mosaic:
        chunk = volume[z_rng, channel, r_rng, c_rng]
    else:
        chunk = volume[z_rng, r_rng, c_rng, channel]

    return chunk


def crop_chunk(chunk, rng, flipped=()):
    """
    Shrink image chunk at volume boundaries, for overall shape consistency.

    Parameters
    ----------
    chunk: ndarray
        sliced image patch

    rng: tuple
        3D index range

    flipped: tuple
        flipped axes

    Returns
    -------
    cropped_chunk: ndarray
        cropped image patch
    """
    # check patch shape and output index ranges
    out_chunk_shape = chunk.shape
    crop_rng = np.zeros(shape=(3,), dtype=int)
    for s in range(3):
        crop_rng[s] = out_chunk_shape[s] - np.arange(rng[s].start, rng[s].stop, rng[s].step).size

    # crop patch if required
    if 0 in flipped:
        cropped_chunk = chunk[crop_rng[0] or None:, ...]
    else:
        cropped_chunk = chunk[:-crop_rng[0] or None, ...]
    if 1 in flipped:
        cropped_chunk = cropped_chunk[:, crop_rng[1] or None:, ...]
    else:
        cropped_chunk = cropped_chunk[:, :-crop_rng[1] or None, ...]
    if 2 in flipped:
        cropped_chunk = cropped_chunk[:, :, crop_rng[2] or None:, ...]
    else:
        cropped_chunk = cropped_chunk[:, :, :-crop_rng[2] or None, ...]

    return cropped_chunk
