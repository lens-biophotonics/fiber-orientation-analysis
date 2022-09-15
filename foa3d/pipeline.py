from os import mkdir, path
from time import perf_counter

import numpy as np

from foa3d.frangi import config_frangi_scales, frangi_filter
from foa3d.input import get_volume_info
from foa3d.odf import compute_scaled_odf, get_sph_harm_ncoeff
from foa3d.output import save_array
from foa3d.preprocessing import correct_image_anisotropy
from foa3d.printing import (color_text, print_analysis_time,
                            print_frangi_heading, print_odf_supervoxel,
                            print_slice_progress, print_slicing_info,
                            print_soma_masking)
from foa3d.slicing import (compute_slice_range, config_frangi_slicing,
                           config_odf_slicing, crop_slice, slice_channel)
from foa3d.utils import (create_background_mask, create_hdf5_file,
                         get_item_bytes, orient_colormap, transform_axes,
                         vector_colormap)


def init_frangi_volumes(volume_shape, slice_shape, resize_ratio, save_dir, volume_name,
                        z_min=0, z_max=None, lpf_soma_mask=False):
    """
    Initialize the output datasets of the Frangi filtering stage.

    Parameters
    ----------
    volume_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    resize_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D resize ratio

    save_dir: str
        saving directory string path

    volume_name: str
        name of the input volume image

    z_min: int
        minimum output z-depth in [px]

    z_max: int
        maximum output z-depth in [px]

    lpf_soma_mask: bool
        neuronal body masking flag

    Returns
    -------
    fiber_vec_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=float32)
        initialized fiber orientation volume

    fiber_vec_colmap: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        initialized orientation colormap image

    frangi_volume: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized Frangi-enhanced image

    fiber_mask: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized fiber mask image

    iso_fiber_volume: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized fiber image (isotropic resolution)

    neuron_mask: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized neuron mask image

    zsel: NumPy slice object
        selected z-depth range

    tmp_hdf5_list: list
        list of dictionaries of temporary HDF5 files (file objects and paths),
        where output data chunks are iteratively stored
    """
    # create Frangi volumes subfolder
    save_dir = path.join(save_dir, 'frangi')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # shape copies
    volume_shape = volume_shape.copy()
    slice_shape = slice_shape.copy()

    # adapt output z-axis shape if required
    if z_min != 0 or z_max is not None:
        if z_max is None:
            z_max = slice_shape[0]
        volume_shape[0] = z_max - z_min
    zsel = slice(z_min, z_max, 1)

    # output datasets shape
    volume_dims = len(volume_shape)
    dset_shape = np.ceil(resize_ratio * volume_shape).astype(int)
    slice_shape[0] = dset_shape[0]

    # create list of temporary files
    tmp_hdf5_list = list()

    # fiber channel
    iso_fiber_path = path.join(save_dir, 'iso_fiber_' + volume_name + '.h5')
    iso_fiber_file, iso_fiber_volume = create_hdf5_file(iso_fiber_path, dset_shape, slice_shape, dtype='uint8')
    tmp_hdf5_list.append({'path': iso_fiber_path, 'obj': iso_fiber_file})

    frangi_path = path.join(save_dir, 'frangi_' + volume_name + '.h5')
    frangi_file, frangi_volume = create_hdf5_file(frangi_path, dset_shape, slice_shape, dtype='uint8')
    tmp_hdf5_list.append({'path': frangi_path, 'obj': frangi_file})

    fiber_mask_path = path.join(save_dir, 'fiber_msk_' + volume_name + '.h5')
    fiber_mask_file, fiber_mask_volume = create_hdf5_file(fiber_mask_path, dset_shape, slice_shape, dtype='uint8')
    tmp_hdf5_list.append({'path': fiber_mask_path, 'obj': fiber_mask_file})

    # neuron channel
    if lpf_soma_mask:
        neuron_mask_path = path.join(save_dir, 'neuron_msk_' + volume_name + '.h5')
        neuron_mask_file, neuron_mask_volume \
            = create_hdf5_file(neuron_mask_path, dset_shape, slice_shape, dtype='uint8')
        tmp_hdf5_list.append({'path': neuron_mask_path, 'obj': neuron_mask_file})
    else:
        neuron_mask_volume = None

    # fiber orientation maps
    vec_dset_shape = tuple(list(dset_shape) + [volume_dims])
    vec_chunk_shape = tuple(list(slice_shape) + [volume_dims])

    fibervec_path = path.join(save_dir, 'fiber_vec_' + volume_name + '.h5')
    _, fiber_vec_volume = create_hdf5_file(fibervec_path, vec_dset_shape, vec_chunk_shape, dtype='float32')

    orientcol_path = path.join(save_dir, 'fiber_cmap_' + volume_name + '.h5')
    orientcol_file, fiber_vec_colmap = create_hdf5_file(orientcol_path, vec_dset_shape, vec_chunk_shape, dtype='uint8')
    tmp_hdf5_list.append({'path': orientcol_path, 'obj': orientcol_file})

    return fiber_vec_volume, fiber_vec_colmap, frangi_volume, fiber_mask_volume, iso_fiber_volume, \
        neuron_mask_volume, zsel, tmp_hdf5_list


def init_odf_volumes(vec_volume_shape, save_dir, odf_degrees=6, odf_scale=15):
    """
    Initialize the output datasets of the ODF analysis stage.

    Parameters
    ----------
    vec_volume_shape: ndarray (shape=(3,), dtype=int)
        vector volume shape [px]

    save_dir: str
        saving directory string path

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    odf_scale: int
        fiber ODF resolution (super-voxel side [px])

    Returns
    -------
    odf_volume: HDF5 dataset (shape=(X,Y,Z,3), dtype=float32)
        initialized dataset of ODF spherical harmonics coefficients

    bg_mrtrix_volume: HDF5 dataset (shape=(X,Y,Z), dtype=uint8)
        initialized background dataset for ODF visualization in Mrtrix3

    odf_tmp_files: list
        list of dictionaries of temporary HDF5 files (file objects and paths)
    """
    # create ODF subfolder
    save_dir = path.join(save_dir, 'odf')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # initialize downsampled background image dataset (HDF5 file)
    bg_shape = np.flip(np.ceil(np.divide(vec_volume_shape, odf_scale))).astype(int)

    bg_tmp_path = path.join(save_dir, 'bg_tmp{}.h5'.format(odf_scale))
    bg_tmp_file, bg_mrtrix_volume \
        = create_hdf5_file(bg_tmp_path, bg_shape, tuple(np.append(bg_shape[:2], 1)), dtype='uint8')
    bg_tmp_dict = {'path': bg_tmp_path, 'obj': bg_tmp_file}

    # initialize ODF dataset
    num_coeff = get_sph_harm_ncoeff(odf_degrees)
    odf_shape = tuple(list(bg_shape) + [num_coeff])
    odf_tmp_path = path.join(save_dir, 'odf_tmp{}.h5'.format(odf_scale))
    odf_tmp_file, odf_volume = create_hdf5_file(odf_tmp_path, odf_shape, (1, 1, 1, num_coeff), dtype='float32')
    odf_tmp_dict = {'path': odf_tmp_path, 'obj': odf_tmp_file}

    # create list of dictionaries of temporary HDF5 files (object and path)
    odf_tmp_files = [bg_tmp_dict, odf_tmp_dict]

    return odf_volume, bg_mrtrix_volume, odf_tmp_files, odf_shape


def iterate_frangi_on_slices(volume, px_size, px_size_iso, smooth_sigma, save_dir, volume_name, max_slice_size=100.0,
                             scales_um=1.25, ch_neuron=0, ch_fiber=1, alpha=0.05, beta=1, gamma=100, dark=False,
                             z_min=0, z_max=None, orient_cmap=False, lpf_soma_mask=False, skeletonize=False,
                             mosaic=False, verbose=True):
    """
    Iteratively apply 3D Frangi filtering to basic TPFM image slices.

    Parameters
    ----------
    volume: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    smooth_sigma: ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (applied to the XY plane)

    save_dir: str
        saving directory string path

    volume_name: str
        name of the input volume image

    max_slice_size: float
        maximum memory size (in bytes) of the basic image slices
        analyzed iteratively

    scales_um: list (dtype=float)
        analyzed spatial scales in [μm]

    ch_neuron: int
        neuronal bodies channel

    ch_fiber: int
        myelinated fibers channel

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    dark: bool
        if True, enhance black 3D tubular structures
        (i.e., negative contrast polarity)

    z_min: int
        minimum output z-depth in [px]

    z_max: int
        maximum output z-depth in [px]

    orient_cmap: bool
        if True, generate color maps based on XY-plane orientation angles
        (instead of using the cartesian components of the estimated vectors)

    lpf_soma_mask: bool
        if True, mask neuronal bodies exploiting the autofluorescence
        signal of lipofuscin pigments

    skeletonize: bool
        if True, apply skeletonization to the boolean mask of myelinated fibers

    mosaic: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    verbose: bool
        verbosity flag

    Returns
    -------
    tmp_hdf5_list: list
        list of temporary file dictionaries
        ('path': file path; 'obj': file object)

    fiber_vec_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=float32)
        fiber orientation vector image

    fiber_vec_colmap: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap image

    frangi_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        Frangi-enhanced volume image (fiber probability volume)

    iso_fiber_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        isotropic fiber image

    fiber_mask_volume: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        fiber mask image

    neuron_mask_volume: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        neuron mask image
    """
    # get info on the input volume image
    volume_shape, volume_shape_um, volume_item_size = get_volume_info(volume, px_size, mosaic=mosaic)

    # get info on the processed image slices
    in_slice_shape, in_slice_shape_um, out_slice_shape, out_volume_shape, px_rsz_ratio, pad = \
        config_frangi_slicing(volume_shape, volume_item_size, px_size, px_size_iso, smooth_sigma,
                              max_slice_size=max_slice_size)

    # initialize the output volume arrays
    fiber_vec_volume, fiber_vec_colmap, frangi_volume, fiber_mask_volume, \
        iso_fiber_volume, neuron_mask_volume, zsel, tmp_hdf5_list = \
        init_frangi_volumes(volume_shape, out_slice_shape, px_rsz_ratio, save_dir, volume_name,
                            z_min=z_min, z_max=z_max, lpf_soma_mask=lpf_soma_mask)

    # compute the Frangi filter's scale values in pixel
    scales_px = config_frangi_scales(scales_um, px_size_iso[0])

    # print info in verbose mode
    if verbose:
        # print Frangi filter configuration
        print_frangi_heading(alpha, beta, gamma, scales_um)

        # print iterative analysis information
        print_slicing_info(volume_shape_um, in_slice_shape_um, px_size, volume_item_size)

        # print neuron masking info
        print_soma_masking(lpf_soma_mask)

    # iteratively apply Frangi filter to basic image slices
    loop_range = np.ceil(np.divide(volume_shape, in_slice_shape)).astype(int)
    total_iter = np.prod(loop_range)
    loop_count = 1
    tic = perf_counter()
    for z in range(loop_range[0]):

        for y in range(loop_range[1]):

            for x in range(loop_range[2]):

                # print progress
                if verbose:
                    print_slice_progress(loop_count, tot=total_iter)

                # index ranges of the analyzed fiber slice (with padding)
                rng_in, pad_mat = compute_slice_range(z, y, x, in_slice_shape, volume_shape, pad_rng=pad)

                # output index ranges
                rng_out, _ = compute_slice_range(z, y, x, out_slice_shape, out_volume_shape)

                # slice fiber image slice
                fiber_mask = slice_channel(volume, rng_in, channel=ch_fiber, mosaic=mosaic)

                # skip background slice
                if np.max(fiber_mask) != 0:

                    # preprocess fiber slice
                    iso_fiber_slice = correct_image_anisotropy(fiber_mask, px_rsz_ratio,
                                                               sigma=smooth_sigma, pad_mat=pad_mat)

                    # crop isotropized fiber slice
                    iso_fiber_slice = crop_slice(iso_fiber_slice, rng_out)

                    # 3D Frangi filter
                    frangi_slice, fiber_vec_slice \
                        = frangi_filter(iso_fiber_slice, scales_px=scales_px,
                                        alpha=alpha, beta=beta, gamma=gamma, dark_fibers=dark)

                    # generate RGB orientation color map
                    if orient_cmap:
                        orientcol_slice = orient_colormap(fiber_vec_slice)
                    else:
                        orientcol_slice = vector_colormap(fiber_vec_slice)

                    # mask background
                    fiber_vec_slice, orientcol_slice, fiber_mask = \
                        mask_background(frangi_slice, fiber_vec_slice, orientcol_slice, thresh_method='li',
                                        skeletonize=skeletonize, invert_mask=False)

                    # (optional) neuronal body masking
                    if lpf_soma_mask:

                        # neuron slice index ranges (without padding)
                        rng_in, _ = compute_slice_range(z, y, x, in_slice_shape, volume_shape)

                        # slice neuron image slice
                        neuron_slice = slice_channel(volume, rng_in, channel=ch_neuron, mosaic=mosaic)

                        # resize neuron slice (lateral downsampling)
                        iso_neuron_slice = correct_image_anisotropy(neuron_slice, px_rsz_ratio)

                        # crop isotropized neuron slice
                        iso_neuron_slice = crop_slice(iso_neuron_slice, rng_out)

                        # mask neuronal bodies
                        fiber_vec_slice, orientcol_slice, neuron_mask = \
                            mask_background(iso_neuron_slice, fiber_vec_slice, orientcol_slice, thresh_method='yen',
                                            skeletonize=False, invert_mask=True)

                        # fill neuron mask
                        neuron_mask_volume[rng_out] = (255 * neuron_mask[zsel, ...]).astype(np.uint8)

                    # fill output volumes
                    vec_rng_out = tuple(np.append(rng_out, slice(0, 3, 1)))
                    fiber_vec_volume[vec_rng_out] = fiber_vec_slice[zsel, ...]
                    fiber_vec_colmap[vec_rng_out] = orientcol_slice[zsel, ...]
                    iso_fiber_volume[rng_out] = iso_fiber_slice[zsel, ...].astype(np.uint8)
                    frangi_volume[rng_out] = (255 * frangi_slice[zsel, ...]).astype(np.uint8)
                    fiber_mask_volume[rng_out] = (255 * (1 - fiber_mask[zsel, ...])).astype(np.uint8)

                # increase loop counter
                loop_count += 1

    # print total filtering time
    if verbose:
        print_analysis_time(tic, total_iter)

    return tmp_hdf5_list, fiber_vec_volume, fiber_vec_colmap, frangi_volume, \
        iso_fiber_volume, fiber_mask_volume, neuron_mask_volume


def iterate_odf_on_slices(fiber_vec_dset, iso_fiber_dset, px_size_iso, save_dir, max_slice_size=100.0, tmp_files=[],
                          odf_scale_um=15, odf_degrees=6, verbose=True):
    """
    Iteratively estimate 3D fiber ODFs over basic orientation data chunks.

    Parameters
    ----------
    fiber_vec_dset: HDF5 dataset (shape=(Z,Y,X,3), dtype=float32)
        fiber orientation vectors dataset

    iso_fiber_dset: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        isotropic fiber volume dataset

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    save_dir: str
        saving directory path

    max_slice_size: float
        maximum memory size (in bytes) of the basic image slices
        analyzed iteratively

    tmp_files: list
        list of dictionaries of temporary HDF5 files (file objects and paths)

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    verbose: bool
        verbosity flag

    Returns
    -------
    odf_volume: HDF5 dataset (shape=(X,Y,Z,3), dtype=float32)
        dataset of ODF spherical harmonics coefficients

    bg_mrtrix_volume: HDF5 dataset (shape=(X,Y,Z), dtype=uint8)
        background dataset for ODF visualization in Mrtrix3

    tmp_files: list
        updated list of dictionaries of temporary HDF5 files
        (file objects and paths)
    """
    # get info on the input volume of orientation vectors
    vec_volume_shape = np.asarray(fiber_vec_dset.shape)[:-1]
    vec_item_size = get_item_bytes(fiber_vec_dset)

    # configure image slicing for ODF analysis
    vec_slice_shape, odf_slice_shape, odf_scale \
        = config_odf_slicing(vec_volume_shape, vec_item_size, px_size_iso,
                             odf_scale_um=odf_scale_um, max_slice_size=max_slice_size)

    # print ODF super-voxel size
    print_odf_supervoxel(vec_slice_shape, px_size_iso, odf_scale_um)

    # initialize ODF analysis output volumes
    odf_volume, bg_mrtrix_volume, odf_tmp_files, odf_volume_shape \
        = init_odf_volumes(vec_volume_shape, save_dir, odf_degrees=odf_degrees, odf_scale=odf_scale)
    tmp_files = tmp_files + odf_tmp_files

    # iteratively apply Frangi filter to basic microscopy image slices
    loop_range = np.ceil(np.divide(vec_volume_shape, vec_slice_shape)).astype(int)
    total_iter = np.prod(loop_range)
    loop_count = 1
    tic = perf_counter()
    for z in range(loop_range[0]):

        for y in range(loop_range[1]):

            for x in range(loop_range[2]):

                # print progress
                if verbose:
                    print_slice_progress(loop_count, tot=total_iter)

                # input index ranges
                rng_in, _ = compute_slice_range(z, y, x, vec_slice_shape, vec_volume_shape)

                # ODF index ranges
                rng_odf, _ = compute_slice_range(x, y, z, np.flip(odf_slice_shape), odf_volume_shape, flip=True)

                # load dataset slices to NumPy arrays, transform axes
                if iso_fiber_dset is None:
                    iso_fiber_slice = None
                else:
                    iso_fiber_slice = iso_fiber_dset[rng_in]
                rng_in = tuple(np.append(rng_in, slice(0, 3, 1)))
                vec_slice = fiber_vec_dset[rng_in]

                # ODF analysis
                odf_slice, bg_mrtrix_slice, \
                    = compute_scaled_odf(odf_scale, vec_slice, iso_fiber_slice, odf_slice_shape, degrees=odf_degrees)

                # transform axes
                odf_slice = transform_axes(odf_slice, swapped=(0, 2), flipped=(0, 1, 2))
                bg_mrtrix_slice = transform_axes(bg_mrtrix_slice, swapped=(0, 2), flipped=(0, 1, 2))

                # crop output slices
                odf_slice = crop_slice(odf_slice, rng_odf, flipped=(0, 1, 2))
                bg_mrtrix_slice = crop_slice(bg_mrtrix_slice, rng_odf, flipped=(0, 1, 2))

                # fill datasets
                bg_mrtrix_volume[rng_odf] = bg_mrtrix_slice
                rng_odf = tuple(np.append(rng_odf, slice(0, odf_slice.shape[-1], 1)))
                odf_volume[rng_odf] = odf_slice

                # increase loop counter
                loop_count += 1

    # print total analysis time
    if verbose:
        print_analysis_time(tic, total_iter)

    return odf_volume, bg_mrtrix_volume, tmp_files


def mask_background(image, fiber_vec_slice, orientcol_slice, thresh_method='yen', skeletonize=False, invert_mask=False):
    """
    Mask orientation volume arrays.

    Parameters
    ----------
    image: numpy.ndarray (shape=(Z,Y,X))
        fiber (or neuron) fluorescence volume image

    fiber_vec_slice: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        fiber orientation vector slice

    orientcol_slice: numpy.ndarray (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap slice

    thresh_method: str
        thresholding method (refer to skimage.filters)

    skeletonize: bool
        mask skeletonization flag

    invert_mask: bool
        mask inversion flag

    Returns
    -------
    fiber_vec_slice: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        orientation vector patch (masked)

    orientcol_slice: numpy.ndarray (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap patch (masked)

    background_mask: numpy.ndarray (shape=(Z,Y,X), dtype=bool)
        background mask
    """
    # generate background mask
    background = create_background_mask(image, thresh_method=thresh_method, skeletonize=skeletonize)

    # invert mask
    if invert_mask:
        background = np.logical_not(background)

    # apply mask to orientation arrays
    fiber_vec_slice[background, :] = 0
    orientcol_slice[background, :] = 0

    return fiber_vec_slice, orientcol_slice, background


def save_frangi_volumes(fiber_vec_volume, fiber_vec_colmap, frangi_volume, fiber_mask, neuron_mask,
                        save_dir, volume_name):
    """
    Save the output arrays of the Frangi filter stage to TIF files.

    Parameters
    ----------
    fiber_vec_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype: float32)
        fiber orientation vector image

    fiber_vec_colmap: HDF5 dataset (shape=(Z,Y,X,3), dtype: uint8)
        orientation colormap image

    frangi_volume: HDF5 dataset (shape=(Z,Y,X), dtype: uint8)
        Frangi-enhanced volume image (fiber probability volume)

    fiber_mask: HDF5 dataset (shape=(Z,Y,X), dtype: uint8)
        fiber mask image

    neuron_mask: HDF5 dataset (shape=(Z,Y,X), dtype: uint8)
        neuron mask image

    save_dir: str
        saving directory string path

    volume_name: str
        name of the input volume image

    Returns
    -------
    None
    """
    # final print
    print(color_text(0, 191, 255, "  Saving Frangi Filter Volumes...\n"))

    # create subfolder
    save_dir = path.join(save_dir, 'frangi')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # save eigenvectors to .npy file
    save_array('fiber_vec_' + volume_name, save_dir, fiber_vec_volume,
               format='npy')

    # save orientation color map to TIF
    save_array('fiber_cmap_' + volume_name, save_dir, fiber_vec_colmap)

    # save Frangi-enhanced fiber volume to TIF
    save_array('frangi_' + volume_name, save_dir, frangi_volume)

    # save masked fiber volume to TIF
    save_array('fiber_msk_' + volume_name, save_dir, fiber_mask)

    # save neuron channel volumes to TIF
    if neuron_mask is not None:
        save_array('neuron_msk_' + volume_name, save_dir, neuron_mask)


def save_odf_volumes(odf_list, bg_mrtrix_list, save_dir, volume_name, odf_scales_um):
    """
    Save the output arrays of the ODF analysis stage to TIF and Nifti files.
    Arrays tagged with 'mrtrixview' are preliminarily transformed
    so that ODF maps viewed in Mrtrix3 are spatially consistent
    with the analyzed microscopy volume, and the output TIF files.

    Parameters
    ----------
    odf_list: list
        list of HDF5 datasets of spherical harmonics coefficients

    bg_mrtrix_list: list
        list of HDF5 datasets of downsampled background images
        for ODF visualization in Mrtrix3 (fiber channel)

    save_dir: str
        saving directory string path

    volume_name: str
        name of the input volume image

    odf_scales_um: list (dtype: float)
        list of fiber ODF resolution values (super-voxel sides [μm])

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    Returns
    -------
    None
    """
    # final print
    print(color_text(0, 191, 255, "  Saving ODF Analysis Volumes...\n\n\n"))

    # create ODF subfolder
    save_dir = path.join(save_dir, 'odf')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # ODF analysis volumes to Nifti files (adjusted view for Mrtrix3)
    for (odf, bg, s) in zip(odf_list, bg_mrtrix_list, odf_scales_um):
        save_array(f'bg_mrtrixview_{s}_' + volume_name, save_dir, bg, format='nii')
        save_array(f'odf_mrtrixview_{s}_' + volume_name, save_dir, odf, format='nii')
