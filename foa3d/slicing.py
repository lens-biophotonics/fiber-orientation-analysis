from itertools import product

import numpy as np
import psutil

from foa3d.utils import get_available_cores


def compute_axis_range(ax, ax_iter, slice_shape, img_shape, slice_per_dim, flip=False, ovlp_rng=0):
    """
    Adjust image slice coordinates at boundaries.

    Parameters
    ----------
    ax: int
        axis index

    ax_iter: int
        iteration counter along axis (see pipeline.iterate_frangi_on_slices)

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slice_per_dim: numpy.ndarray (shape=(3,), dtype=int)
        number of image slices along each axis

    flip: bool
        if True, flip axis coordinates

    ovlp_rng: int
        optional slice sampling range
        extension along each axis (on each side)

    Returns
    -------
    start: int
        adjusted start index

    stop: int
        adjusted stop index

    ovlp: numpy.ndarray (shape=(2,), dtype=int)
        lower and upper overlapped boundaries
        along axis
    """

    # compute start and stop coordinates
    start = ax_iter[ax] * slice_shape[ax]
    stop = start + slice_shape[ax]

    # flip coordinates if required
    if flip:
        start_tmp = start
        start = img_shape[ax] - stop
        stop = img_shape[ax] - start_tmp

    # adjust start and stop coordinates
    start, stop, ovlp = trim_axis_range(img_shape, start, stop, ax=ax, ovlp_rng=ovlp_rng)

    # handle image shape residuals at boundaries
    if ax_iter[ax] == slice_per_dim[ax] - 1:
        if np.remainder(img_shape[ax], slice_shape[ax]) > 0:
            stop = img_shape[ax]
            ovlp[1] = 0

    return start, stop, ovlp


def compute_slice_range(ax_iter, slice_shape, img_shape, slice_per_dim, ovlp_rng=0, flip=False):
    """
    Compute basic slice coordinates from microscopy volume image.

    Parameters
    ----------
    ax_iter: tuple
        iteration counters along axes

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slice_per_dim: numpy.ndarray (shape=(3,), dtype=int)
        number of image slices along each axis

    ovlp_rng: int
        optional slice sampling range
        extension along each axis (on each side)

    flip: bool
        if True, flip axes coordinates

    Returns
    -------
    rng: tuple
        3D slice index ranges

    ovlp: np.ndarray (shape=(3,2), dtype=int)
        overlapped boundaries
    """

    # adjust original image patch coordinates
    # and generate padding range matrix
    dims = len(ax_iter)
    start = np.zeros((dims,), dtype=int)
    stop = np.zeros((dims,), dtype=int)
    ovlp = np.zeros((dims, 2), dtype=int)
    slc = tuple()
    for ax in range(dims):
        start[ax], stop[ax], ovlp[ax] = \
            compute_axis_range(ax, ax_iter, slice_shape, img_shape, slice_per_dim, flip=flip, ovlp_rng=ovlp_rng)
        slc += (slice(start[ax], stop[ax], 1),)

    # generate tuple of slice index ranges
    rng = np.index_exp[slc]

    # handle invalid slices
    for r in rng:
        if r.start is None:
            return None, ovlp

    return rng, ovlp


def compute_slice_shape(img_shape, item_size, px_size=None, max_size=100, ovlp_rng=0):
    """
    Compute basic image chunk shape depending on its maximum size (in bytes).

    Parameters
    ----------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_size: int
        image item size (in bytes)

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    max_slice_size: float
        maximum memory size (in bytes) of the basic image slices
        analyzed using parallel threads

    ovlp_rng: int
        slice overlap range along each axis

    Returns
    -------
    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the basic image slices
        analyzed using parallel threads [μm]
        (if px_size is provided)
    """
    if len(img_shape) == 4:
        item_size = item_size * 3

    slice_depth = img_shape[0]
    slice_side = np.round(1024 * np.sqrt((max_size / (slice_depth * item_size))) - 2 * ovlp_rng)
    slice_shape = np.array([slice_depth, slice_side, slice_side]).astype(int)
    slice_shape = np.min(np.stack((img_shape[:3], slice_shape)), axis=0)

    if px_size is not None:
        slice_shape_um = np.multiply(slice_shape, px_size)
        return slice_shape, slice_shape_um

    else:
        return slice_shape


def compute_overlap_range(smooth_sigma, frangi_sigma, truncate=4):
    """
    Compute lateral slice extension range
    for coping with smoothing-related boundary artifacts.

    Parameters
    ----------
    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (applied to the XY plane)

    frangi_sigma: numpy.ndarray (dtype=int)
        analyzed spatial scales [px]

    truncate: int
        neglect the Gaussian smoothing kernel
        after this many standard deviations

    Returns
    -------
    ext_rng: int
        slice extended index range (on each side)
    """
    max_sigma = np.max(np.concatenate((smooth_sigma, frangi_sigma)))

    ext_rng = int(np.ceil(2 * truncate * max_sigma) // 2 * 2 + 1) if smooth_sigma is not None else 0

    return ext_rng


def config_frangi_batch(frangi_scales, mem_growth_factor=149.7, mem_fudge_factor=1.0,
                        min_slice_size_mb=-1, jobs=None, max_ram_mb=None):
    """
    Compute size and number of the batches of basic microscopy image slices
    analyzed in parallel.

    Parameters
    ----------
    frangi_scales: list (dtype=float)
        analyzed spatial scales in [μm]

    mem_growth_factor: float
        empirical memory growth factor
        of the Frangi filtering stage

    mem_fudge_factor: float
        memory fudge factor

    min_slice_size_mb: float
        minimum slice size in [MB]

    jobs: int
        number of parallel jobs (threads)
        used by the Frangi filtering stage

    max_ram_mb: float
        maximum RAM available to the Frangi filtering stage [MB]

    Returns
    -------
    slice_batch_size: int
        slice batch size

    slice_size_mb: float
        memory size (in megabytes) of the basic image slices
        fed to the Frangi filter
    """
    # maximum RAM not provided: use all
    if max_ram_mb is None:
        max_ram_mb = psutil.virtual_memory()[1] / 1e6

    # number of logical cores
    num_cpu = get_available_cores()
    if jobs is None:
        jobs = num_cpu

    # number of spatial scales
    num_scales = len(frangi_scales)

    # initialize slice batch size
    slice_batch_size = np.min([jobs // num_scales, num_cpu]).astype(int)
    if slice_batch_size == 0:
        slice_batch_size = 1

    # get image slice size
    slice_size_mb = get_slice_size(max_ram_mb, mem_growth_factor, mem_fudge_factor, slice_batch_size, num_scales)
    while slice_size_mb < min_slice_size_mb:
        slice_batch_size -= 1
        slice_size_mb = get_slice_size(max_ram_mb, mem_growth_factor, mem_fudge_factor, slice_batch_size, num_scales)

    return slice_batch_size, slice_size_mb


def config_frangi_slicing(img_shape, item_size, px_size, px_size_iso, smooth_sigma, frangi_sigma, lpf_soma_mask,
                          batch_size, slice_size):
    """
    Image slicing configuration for the parallel Frangi filtering of basic chunks
    of the input microscopy volume using concurrent threads.

    Parameters
    ----------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_size: int
        image item size (in bytes)

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    frangi_sigma: numpy.ndarray (or int)
        Frangi filter scales [px]

    batch_size: int
        slice batch size

    slice_size: float
        maximum memory size (in megabytes) of the basic image slices
        analyzed iteratively

    Returns
    -------
    in_rng_lst: list
        list of input slice index ranges

    in_rng_lst_neu: list
        list of soma channel slice index ranges

    out_rng_lst: list
        list of output slice index ranges

    in_ovlp_lst: list
        list of slice overlap arrays

    in_slice_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the basic image slices analyzed iteratively [μm]

    out_slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the processed image slices [px]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    tot_slice_num: int
        total number of analyzed image slices

    batch_size: int
        adjusted slice batch size
    """

    # compute input patch padding range (border artifacts suppression)
    ovlp_rng = compute_overlap_range(smooth_sigma, frangi_sigma)

    # shape of processed TPFM slices
    in_slice_shape, in_slice_shape_um = compute_slice_shape(img_shape, item_size,
                                                            px_size=px_size, max_size=slice_size, ovlp_rng=ovlp_rng)

    # adjust shapes according to the anisotropy correction
    px_rsz_ratio = np.divide(px_size, px_size_iso)
    out_slice_shape = np.ceil(px_rsz_ratio * in_slice_shape).astype(int)
    out_image_shape = np.ceil(px_rsz_ratio * img_shape).astype(int)

    # iteratively define input/output slice 3D ranges
    slice_per_dim = np.ceil(np.divide(img_shape, in_slice_shape)).astype(int)
    tot_slice_num = int(np.prod(slice_per_dim))

    # initialize empty range lists
    out_rng_lst = list()
    in_ovlp_lst = list()
    in_rng_lst = list()
    in_rng_lst_neu = list()
    for z, y, x in product(range(slice_per_dim[0]), range(slice_per_dim[1]), range(slice_per_dim[2])):

        # index ranges of analyzed fiber image slices (with padding)
        in_rng, ovlp = compute_slice_range((z, y, x), in_slice_shape, img_shape, slice_per_dim, ovlp_rng=ovlp_rng)

        # get output slice ranges
        # for valid input range instances
        if in_rng is not None:
            in_rng_lst.append(in_rng)
            in_ovlp_lst.append(ovlp)

            # output index ranges
            out_rng, _ = compute_slice_range((z, y, x), out_slice_shape, out_image_shape, slice_per_dim)
            out_rng_lst.append(out_rng)

            # optional neuron masking
            if lpf_soma_mask:
                in_rng_neu, _ = compute_slice_range((z, y, x), in_slice_shape, img_shape, slice_per_dim)
                in_rng_lst_neu.append(in_rng_neu)
            else:
                in_rng_lst_neu.append(None)
        else:
            tot_slice_num -= 1

    # adjust slice batch size
    if batch_size > tot_slice_num:
        batch_size = tot_slice_num

    return in_rng_lst, in_rng_lst_neu, out_rng_lst, in_ovlp_lst, \
        in_slice_shape_um, out_slice_shape, px_rsz_ratio, tot_slice_num, batch_size


def crop_slice(img_slice, rng, ovlp=None, flipped=()):
    """
    Shrink image slice at volume boundaries, for overall shape consistency.

    Parameters
    ----------
    img_slice: numpy.ndarray (axis order: (Z,Y,X))
        image slice

    rng: tuple
        3D slice index range

    ovlp: np.ndarray (shape=(3,2), dtype=int)
        overlapped boundaries

    flipped: tuple
        flipped axes

    Returns
    -------
    cropped: numpy.ndarray
        cropped image slice
    """

    # delete overlapping slice boundaries
    if ovlp is not None:
        img_slice = img_slice[ovlp[0, 0]:img_slice.shape[0] - ovlp[0, 1],
                              ovlp[1, 0]:img_slice.shape[1] - ovlp[1, 1],
                              ovlp[2, 0]:img_slice.shape[2] - ovlp[2, 1]]

    # check slice shape and output index ranges
    out_slice_shape = img_slice.shape
    crop_rng = np.zeros(shape=(3,), dtype=int)
    for s in range(3):
        crop_rng[s] = out_slice_shape[s] - np.arange(rng[s].start, rng[s].stop, rng[s].step).size

    # crop slice if required
    cropped = \
        img_slice[crop_rng[0] or None:, ...] if 0 in flipped else img_slice[:-crop_rng[0] or None, ...]
    cropped = \
        cropped[:, crop_rng[1] or None:, ...] if 1 in flipped else cropped[:, :-crop_rng[1] or None, ...]
    cropped = \
        cropped[:, :, crop_rng[2] or None:, ...] if 2 in flipped else cropped[:, :, :-crop_rng[2] or None, ...]

    return cropped


def get_slice_size(max_ram, mem_growth_factor, mem_fudge_factor, slice_batch_size, num_scales=1):
    """
    Compute the size of the basic microscopy image slices fed to the Frangi filtering stage.

    Parameters
    ----------
    max_ram: float
        available RAM

    mem_growth_factor: float
        empirical memory growth factor
        of the pipeline stage (Frangi filter or ODF estimation)

    mem_fudge_factor: float
        memory fudge factor

    slice_batch_size: int
        slice batch size

    num_scales: int
        number of spatial scales analyzed in parallel

    Returns
    -------
    slice_size: float
        memory size (in megabytes) of the basic image slices
        fed to the pipeline stage
    """
    slice_size = max_ram / (slice_batch_size * mem_growth_factor * mem_fudge_factor * num_scales)

    return slice_size


def slice_channel(img, rng, channel, mosaic=False):
    """
    Slice desired channel from input image volume.

    Parameters
    ----------
    img: numpy.ndarray
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

    if channel is None:
        image_slice = img[z_rng, r_rng, c_rng]
    else:
        if mosaic:
            image_slice = img[z_rng, channel, r_rng, c_rng]
        else:
            image_slice = img[z_rng, r_rng, c_rng, channel]

    return image_slice


def trim_axis_range(img_shape, start, stop, ax, ovlp_rng=0):
    """
    Trim slice axis range at total image boundaries.

    Parameters
    ----------
    img_shape: tuple or np.ndarray
        image shape

    start: int
        slice start coordinate

    stop: int
        slice stop coordinate

    ax: int
        image axis

    ovlp_rng: int
        optional slice sampling range
        extension along each axis (on each side)

    Returns
    -------
    start: int
        adjusted slice start coordinate

    stop: int
        adjusted slice stop coordinate

    ovlp: numpy.ndarray (shape=(2,), dtype=int)
        lower and upper overlap ranges
    """

    # initialize slice overlap array
    ovlp = np.zeros((2,), dtype=int)

    # skip z-axis
    if ax != 0:

        # adjust start coordinate
        start -= ovlp_rng
        if start < 0:
            ovlp[0] = ovlp_rng + start
            start = 0
        else:
            ovlp[0] = ovlp_rng

        # adjust stop coordinate
        stop += ovlp_rng
        if stop > img_shape[ax]:
            ovlp[1] = img_shape[ax] - (stop - ovlp_rng)
            stop = img_shape[ax]
        else:
            ovlp[1] = ovlp_rng

    return start, stop, ovlp
