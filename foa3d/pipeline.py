from os import mkdir, path
from time import perf_counter

import numpy as np

from foa3d.frangi import config_frangi_scales, frangi_filter
from foa3d.input import get_volume_info
from foa3d.odf import compute_scaled_odf, get_sph_harm_ncoeff
from foa3d.output import save_array
from foa3d.preprocessing import correct_image_anisotropy
from foa3d.printing import (colored, print_analysis_time, print_frangi_heading,
                            print_masking_info, print_odf_supervoxel,
                            print_slice_progress, print_slicing_info)
from foa3d.slicing import (compute_chunk_range, config_frangi_slicing,
                           config_odf_slicing, crop_chunk, slice_channel)
from foa3d.utils import (create_background_mask, create_hdf5_file,
                         get_item_bytes, orient_colormap, transform_axes,
                         vector_colormap)


def init_frangi_volumes(volume_shape, chunk_shape, resize_ratio, save_dir, volume_name,
                        z_min=0, z_max=None, lpf_soma_mask=False):
    """
    Initialize the output datasets of the Frangi filtering stage.

    Parameters
    ----------
    volume_shape: ndarray (shape=(3,), dtype=int)
        volume shape [px]

    chunk_shape: ndarray (shape=(3,), dtype=int)
        basic output chunk shape [px]

    resize_ratio: ndarray (shape=(3,), dtype=float)
        3D axes resize ratio

    save_dir: str
        saving directory string path

    volume_name: str
        name of the input TPFM volume

    z_min: int
        minimum output z-depth in [px]

    z_max: int
        maximum output z-depth in [px]

    lpf_soma_mask: bool
        neuronal body masking flag

    Returns
    -------
    eigenvec_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=float32)
        initialized orientation vector image

    orientcol_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        initialized orientation colormap image

    frangi_volume: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized Frangi-enhanced image

    fiber_mask: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized fiber mask image

    iso_fiber_volume: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized fiber image

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
    chunk_shape = chunk_shape.copy()

    # adapt output z-axis shape if required
    if z_min != 0 or z_max is not None:
        if z_max is None:
            z_max = chunk_shape[0]
        volume_shape[0] = z_max - z_min
    zsel = slice(z_min, z_max, 1)

    # output datasets shape
    volume_dims = len(volume_shape)
    dset_shape = np.ceil(resize_ratio * volume_shape).astype(int)
    chunk_shape[0] = dset_shape[0]

    # create list of temporary files
    tmp_hdf5_list = list()

    # fiber channel
    iso_fiber_path = path.join(save_dir, 'iso_fiber_' + volume_name + '.h5')
    iso_fiber_file, iso_fiber_volume = create_hdf5_file(iso_fiber_path, dset_shape, chunk_shape, dtype='uint8')
    tmp_hdf5_list.append({'path': iso_fiber_path, 'obj': iso_fiber_file})

    frangi_path = path.join(save_dir, 'frangi_' + volume_name + '.h5')
    frangi_file, frangi_volume = create_hdf5_file(frangi_path, dset_shape, chunk_shape, dtype='uint8')
    tmp_hdf5_list.append({'path': frangi_path, 'obj': frangi_file})

    fiber_mask_path = path.join(save_dir, 'fiber_msk_' + volume_name + '.h5')
    fiber_mask_file, fiber_mask_volume = create_hdf5_file(fiber_mask_path, dset_shape, chunk_shape, dtype='uint8')
    tmp_hdf5_list.append({'path': fiber_mask_path, 'obj': fiber_mask_file})

    # neuron channel
    if lpf_soma_mask:
        neuron_mask_path = path.join(save_dir, 'neuron_msk_' + volume_name + '.h5')
        neuron_mask_file, neuron_mask_volume \
            = create_hdf5_file(neuron_mask_path, dset_shape, chunk_shape, dtype='uint8')
        tmp_hdf5_list.append({'path': neuron_mask_path, 'obj': neuron_mask_file})
    else:
        neuron_mask_volume = None

    # fiber orientation maps
    vec_dset_shape = tuple(list(dset_shape) + [volume_dims])
    vec_chunk_shape = tuple(list(chunk_shape) + [volume_dims])

    eigenvec_path = path.join(save_dir, 'evi_' + volume_name + '.h5')
    _, eigenvec_volume = create_hdf5_file(eigenvec_path, vec_dset_shape, vec_chunk_shape, dtype='float32')

    orientcol_path = path.join(save_dir, 'cmap_' + volume_name + '.h5')
    orientcol_file, orientcol_volume = create_hdf5_file(orientcol_path, vec_dset_shape, vec_chunk_shape, dtype='uint8')
    tmp_hdf5_list.append({'path': orientcol_path, 'obj': orientcol_file})

    return eigenvec_volume, orientcol_volume, frangi_volume, fiber_mask_volume, iso_fiber_volume, \
        neuron_mask_volume, zsel, tmp_hdf5_list


def iterate_frangi_on_slices(volume, px_size, px_size_iso, smooth_sigma, save_dir, volume_name, max_slice_size=100.0,
                             scales_um=1.25, ch_neuron=0, ch_fiber=1, alpha=0.05, beta=1, gamma=100, dark_fibers=False,
                             z_min=0, z_max=None, orient_cmap=False, lpf_soma_mask=False, skeletonize=False,
                             mosaic=False, verbose=True):
    """
    Iteratively apply 3D Frangi filtering to basic TPFM image slices.

    Parameters
    ----------
    volume: ndarray (shape=(Z,Y,X))
        input TPFM image volume

    px_size: ndarray (shape=(3,), dtype=float)
        original TPFM pixel size in [μm]

    px_size_iso: ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size in [μm]

    smooth_sigma: ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (applied to the XY plane)

    save_dir: str
        saving directory string path

    volume_name: str
        name of the input TPFM volume

    max_slice_MB: float
        maximum memory size (in bytes) of the basic slices analyzed iteratively

    scales_um: list (dtype=float)
        analyzed spatial scales in [μm]

    ch_neuron: int
        neuron fluorescence channel (default: 0)

    ch_fiber: int
        fiber fluorescence channel (default: 1)

    alpha: float
        plate-like score sensitivity (default: 0.001)

    beta: float
        blob-like score sensitivity (default: 1)

    gamma: float
        background score sensitivity (default: automatic)

    dark_fibers: bool
        if True, enhance black 3D tubular structures

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

    vec_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=float32)
        orientation vector image

    orientcol_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap image

    frangi_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        Frangi filter-enhanced image

    iso_fiber_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        isotropic fiber image

    fiber_mask_volume: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        fiber mask image

    neuron_mask_volume: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        neuron mask image

    out_chunk_shape: ndarray (shape=(3,), dtype=int)
        shape of the processed image patches [px]
    """
    # get info on the input TPFM image volume
    volume_shape, volume_shape_um, volume_item_size = get_volume_info(volume, px_size, mosaic=mosaic)

    # get info on the processed TPFM slices
    in_chunk_shape, in_chunk_shape_um, out_chunk_shape, out_volume_shape, px_rsz_ratio, pad \
        = config_frangi_slicing(volume_shape, volume_item_size, px_size, px_size_iso, smooth_sigma,
                                max_slice_size=max_slice_size)

    # initialize the output volume arrays
    vec_volume, orientcol_volume, frangi_volume, fiber_mask_volume, \
        iso_fiber_volume, neuron_mask_volume, zsel, tmp_hdf5_list \
        = init_frangi_volumes(volume_shape, out_chunk_shape, px_rsz_ratio, save_dir, volume_name,
                              z_min=z_min, z_max=z_max, lpf_soma_mask=lpf_soma_mask)

    # compute the Frangi filter's scale values in pixel
    sigma_px = config_frangi_scales(scales_um, px_size_iso[0])

    # print info in verbose mode
    if verbose:
        # print Frangi filter configuration
        print_frangi_heading(alpha, beta, gamma, scales_um)

        # print iterative analysis information
        print_slicing_info(volume_shape_um, in_chunk_shape_um, px_size, volume_item_size)

        # print neuron masking info
        print_masking_info(lpf_soma_mask)

    # iteratively apply Frangi filtering to basic TPFM slices
    loop_range = np.ceil(np.divide(volume_shape, in_chunk_shape)).astype(int)
    total_iter = np.prod(loop_range)
    loop_count = 1
    tic = perf_counter()
    for z in range(loop_range[0]):

        for y in range(loop_range[1]):

            for x in range(loop_range[2]):

                # print progress
                if verbose:
                    print_slice_progress(loop_count, tot=total_iter)

                # index ranges of the analyzed fiber patch (with padding)
                rng_in, pad_mat = compute_chunk_range(z, y, x, in_chunk_shape, volume_shape, pad_rng=pad)

                # output index ranges
                rng_out, _ = compute_chunk_range(z, y, x, out_chunk_shape, out_volume_shape)

                # slice fiber image patch
                fiber_mask = slice_channel(volume, rng_in, channel=ch_fiber, mosaic=mosaic)

                # skip background patch
                if np.max(fiber_mask) != 0:

                    # preprocess fiber patch
                    iso_fiber_patch = correct_image_anisotropy(fiber_mask, px_rsz_ratio,
                                                               sigma=smooth_sigma, pad_mat=pad_mat)

                    # crop isotropized fiber patch
                    iso_fiber_patch = crop_chunk(iso_fiber_patch, rng_out)

                    # 3D Frangi filtering
                    frangi_patch, vec_patch \
                        = frangi_filter(iso_fiber_patch, sigma_px=sigma_px,
                                        alpha=alpha, beta=beta, gamma=gamma, dark_fibers=dark_fibers)

                    # generate RGB orientation color map
                    if orient_cmap:
                        orientcol_patch = orient_colormap(vec_patch)
                    else:
                        orientcol_patch = vector_colormap(vec_patch)

                    # mask background
                    vec_patch, orientcol_patch, fiber_mask = \
                        mask_background(frangi_patch, vec_patch, orientcol_patch, thresh_method='li',
                                        skeletonize=skeletonize, invert_mask=False)

                    # (optional) neuronal body masking
                    if lpf_soma_mask:

                        # neuron patch index ranges (without padding)
                        rng_in, _ = compute_chunk_range(z, y, x, in_chunk_shape, volume_shape)

                        # slice neuron image patch
                        neuron_patch = slice_channel(volume, rng_in, channel=ch_neuron, mosaic=mosaic)

                        # resize neuron patch (lateral downsampling)
                        iso_neuron_patch = correct_image_anisotropy(neuron_patch, px_rsz_ratio)

                        # crop isotropized neuron patch
                        iso_neuron_patch = crop_chunk(iso_neuron_patch, rng_out)

                        # mask neuronal bodies
                        vec_patch, orientcol_patch, neuron_mask = \
                            mask_background(iso_neuron_patch, vec_patch, orientcol_patch, thresh_method='yen',
                                            skeletonize=False, invert_mask=True)

                        # fill neuron mask
                        neuron_mask_volume[rng_out] = (255 * neuron_mask[zsel, ...]).astype(np.uint8)

                    # fill output volumes
                    vec_rng_out = tuple(np.append(rng_out, slice(0, 3, 1)))
                    vec_volume[vec_rng_out] = vec_patch[zsel, ...]
                    orientcol_volume[vec_rng_out] = orientcol_patch[zsel, ...]
                    iso_fiber_volume[rng_out] = iso_fiber_patch[zsel, ...].astype(np.uint8)
                    frangi_volume[rng_out] = (255 * frangi_patch[zsel, ...]).astype(np.uint8)
                    fiber_mask_volume[rng_out] = (255 * (1 - fiber_mask[zsel, ...])).astype(np.uint8)

                # increase loop counter
                loop_count += 1

    # print total filtering time
    if verbose:
        print_analysis_time(tic, total_iter)

    return tmp_hdf5_list, vec_volume, orientcol_volume, frangi_volume, \
        iso_fiber_volume, fiber_mask_volume, neuron_mask_volume


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
        fiber ODF resolution (super-voxel side in [px])

    Returns
    -------
    odf_volume: HDF5 dataset (shape=(X,Y,Z,3), dtype=float32)
        initialized dataset of the ODF spherical harmonics coefficients

    bg_mrtrix_volume: HDF5 dataset (shape=(X,Y,Z), dtype=uint8)
        initialized dataset of the ODF background image for visualization
        in Mrtrix3

    odf_tmp_files: list
        list of dictionaries of temporary HDF5 files
        (file objects and paths)
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


def iterate_odf_on_slices(vec_dset, iso_fiber_dset, px_size_iso, save_dir, max_slice_size=100.0, tmp_files=[],
                          odf_scale_um=15, odf_degrees=6, verbose=True):
    """
    Iteratively estimate 3D fiber ODFs over basic orientation data chunks.

    Parameters
    ----------
    vec_dset: HDF5 dataset (shape=(Z,Y,X,3), dtype=float32)
        orientation vectors dataset

    iso_fiber_dset: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        isotropic fiber volume dataset

    px_size_iso: ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size in [μm]

    save_dir: str
        saving directory path

    max_slice_MB: float
        maximum memory size (in bytes) of the basic slices analyzed iteratively

    tmp_files: list
        list of dictionaries of temporary HDF5 files (file objects and paths)

    odf_scale_um: float
        fiber ODF resolution (super-voxel side in [μm])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    verbose: bool
        verbosity flag

    Returns
    -------
    odf_volume: HDF5 dataset (shape=(X,Y,Z,3), dtype=float32)
        spherical harmonics coefficients of 3D fiber ODF

    bg_mrtrix_volume: HDF5 dataset (shape=(X,Y,Z), dtype=uint8)
        ODF background image for ODF visualization in Mrtrix3

    tmp_files: list
        updated list of dictionaries of temporary HDF5 files
        (file objects and paths)
    """
    # get info on the input volume of orientation vectors
    vec_volume_shape = np.asarray(vec_dset.shape)[:-1]
    vec_item_size = get_item_bytes(vec_dset)

    # configure image slicing for ODF analysis
    vec_patch_shape, odf_patch_shape, odf_scale \
        = config_odf_slicing(vec_volume_shape, vec_item_size, px_size_iso,
                             odf_scale_um=odf_scale_um, max_slice_size=max_slice_size)

    # print ODF super-voxel size
    print_odf_supervoxel(vec_patch_shape, px_size_iso, odf_scale_um)

    # initialize ODF analysis output volumes
    odf_volume, bg_mrtrix_volume, odf_tmp_files, odf_volume_shape \
        = init_odf_volumes(vec_volume_shape, save_dir, odf_degrees=odf_degrees, odf_scale=odf_scale)
    tmp_files = tmp_files + odf_tmp_files

    # iteratively apply Frangi filtering to basic TPFM slices
    loop_range = np.ceil(np.divide(vec_volume_shape, vec_patch_shape)).astype(int)
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
                rng_in, _ = compute_chunk_range(z, y, x, vec_patch_shape, vec_volume_shape)

                # ODF index ranges
                rng_odf, _ = compute_chunk_range(x, y, z, np.flip(odf_patch_shape), odf_volume_shape, flip=True)

                # load dataset chunks to NumPy arrays, transform axes
                if iso_fiber_dset is None:
                    iso_fiber_chunk = None
                else:
                    iso_fiber_chunk = iso_fiber_dset[rng_in]
                rng_in = tuple(np.append(rng_in, slice(0, 3, 1)))
                vec_chunk = vec_dset[rng_in]

                # ODF analysis
                odf_chunk, bg_mrtrix_chunk, \
                    = compute_scaled_odf(odf_scale, vec_chunk, iso_fiber_chunk, odf_patch_shape, degrees=odf_degrees)

                # transform axes
                odf_chunk = transform_axes(odf_chunk, swapped=(0, 2), flipped=(0, 1, 2))
                bg_mrtrix_chunk = transform_axes(bg_mrtrix_chunk, swapped=(0, 2), flipped=(0, 1, 2))

                # crop output chunks
                odf_chunk = crop_chunk(odf_chunk, rng_odf, flipped=(0, 1, 2))
                bg_mrtrix_chunk = crop_chunk(bg_mrtrix_chunk, rng_odf, flipped=(0, 1, 2))

                # fill datasets
                bg_mrtrix_volume[rng_odf] = bg_mrtrix_chunk
                rng_odf = tuple(np.append(rng_odf, slice(0, odf_chunk.shape[-1], 1)))
                odf_volume[rng_odf] = odf_chunk

                # increase loop counter
                loop_count += 1

    # print total analysis time
    if verbose:
        print_analysis_time(tic, total_iter)

    return odf_volume, bg_mrtrix_volume, tmp_files


def mask_background(image, vec_patch, orientcol_patch, thresh_method='yen', skeletonize=False, invert_mask=False):
    """
    Mask orientation volume arrays.

    Parameters
    ----------
    image: ndarray (shape=(Z,Y,X))
        fiber (or neuron) fluorescence image volume

    vec_patch: ndarray (shape=(Z,Y,X,3), dtype=float)
        orientation vector patch

    orientcol_patch: ndarray (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap patch

    thresh_method: str
        thresholding method (refer to skimage.filters)

    skeletonize: bool
        mask skeletonization flag

    invert_mask: bool
        mask inversion flag

    Return
    ------
    eigenvec_patch: ndarray
        orientation vector patch (masked)

    orientcol_patch: ndarray
        orientation colormap patch (masked)

    background_mask: ndarray (dtype: bool)
        background mask
    """
    # generate background mask
    background = create_background_mask(image, thresh_method=thresh_method, skeletonize=skeletonize)

    # invert mask
    if invert_mask:
        background = np.logical_not(background)

    # apply mask to orientation arrays
    vec_patch[background, :] = 0
    orientcol_patch[background, :] = 0

    return vec_patch, orientcol_patch, background


def save_frangi_volumes(vec_volume, vec_colmap, frangi_volume, fiber_mask, neuron_mask, save_dir, volume_name):
    """
    Save the output arrays of the Frangi filtering stage to TIF files.

    Parameters
    ----------
    vec_volume: HDF5 dataset (shape=(Z,Y,X,3), dtype: float32)
        dominant eigenvector volume

    vec_colmap: HDF5 dataset (shape=(Z,Y,X,C), dtype: uint8)
        orientation colormap volume

    frangi_volume: HDF5 dataset (shape=(Z,Y,X), dtype: uint8)
        Frangi-enhanced image volume (vesselness probability)

    fiber_mask: HDF5 dataset (shape=(Z,Y,X), dtype: uint8)
        fiber volume mask

    neuron_mask: HDF5 dataset (shape=(Z,Y,X), dtype: uint8)
        neuronal bodies volume mask

    save_dir: string
        saving directory

    volume_name: string
        TPFM image volume name

    Returns
    -------
    None
    """
    # final print
    print(colored(0, 191, 255, "  Saving Frangi Filtering Volumes...\n"))

    # create subfolder
    save_dir = path.join(save_dir, 'frangi')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # save eigenvectors to .npy file
    save_array('evi_' + volume_name, save_dir, vec_volume,
               format='npy')

    # save orientation color map to TIF
    save_array('cmap_' + volume_name, save_dir, vec_colmap)

    # save Frangi-enhanced fiber volume to TIF
    save_array('frangi_' + volume_name, save_dir, frangi_volume)

    # save masked fiber volume to TIF
    save_array('fiber_mask_' + volume_name, save_dir, fiber_mask)

    # save neuron channel volumes to TIF
    if neuron_mask is not None:
        save_array('neuron_mask_' + volume_name, save_dir, neuron_mask)


def save_odf_volumes(odf_list, bg_mrtrix_list, save_dir, volume_name, odf_scales_um):
    """
    Save the output arrays of the ODF analysis stage to TIF and Nifti files.\n
    NOTE: arrays tagged with 'mrtrixview' are preliminarily transformed
               so that ODF maps viewed in Mrtrix3 are spatially consistent
               with the analyzed microscopy volume, and the output TIF files.

    Parameters
    ----------
    odf_list: list
        list of HDF5 datasets of spherical harmonics coefficients

    bg_mrtrix_list: list
        list of HDF5 datasets of downsampled background images
        for ODF visualization in Mrtrix3 (fiber channel)

    save_dir: string
        saving directory

    volume_name: string
        input TPFM image volume name

    odf_scales_um: list (dtype: float)
        list of fiber ODF resolution values (super-voxel sides in [μm])

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
    print(colored(0, 191, 255, "  Saving ODF Analysis Volumes...\n\n\n"))

    # create ODF subfolder
    save_dir = path.join(save_dir, 'odf')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # odf analysis volumes to Nifti files (adjusted view for Mrtrix3)
    for (odf, bg, s) in zip(odf_list, bg_mrtrix_list, odf_scales_um):
        save_array(f'bg_mrtrixview_{s}_' + volume_name, save_dir, bg, format='nii')
        save_array(f'odf_mrtrixview_{s}_' + volume_name, save_dir, odf, format='nii')
