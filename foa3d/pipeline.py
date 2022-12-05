from os import mkdir, path

import numpy as np
from alive_progress import alive_bar
from numba import njit

from foa3d.frangi import config_frangi_scales, frangi_filter
from foa3d.input import get_image_info
from foa3d.odf import compute_scaled_odf, get_sph_harm_ncoeff
from foa3d.output import save_array
from foa3d.preprocessing import correct_image_anisotropy
from foa3d.printing import (color_text, print_frangi_heading,
                            print_odf_supervoxel, print_slicing_info,
                            print_soma_masking)
from foa3d.slicing import (compute_slice_range, config_frangi_slicing,
                           config_odf_slicing, crop_slice, slice_channel)
from foa3d.utils import (create_background_mask, create_hdf5_file,
                         get_item_bytes, orient_colormap, transform_axes,
                         vector_colormap)


@njit(cache=True)
def compute_fractional_anisotropy(eigenval):
    """
    Compute structure tensor fractional anisotropy
    as in Schilling et al. (2018).

    Parameters
    ----------
    eigenval: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        structure tensor eigenvalues (best local spatial scale)

    Returns
    -------
    frac_anis: numpy.ndarray (shape=(3,), dtype=float)
        fractional anisotropy
    """
    frac_anis = \
        np.sqrt(0.5 * np.divide((eigenval[..., 0] - eigenval[..., 1]) ** 2 +
                                (eigenval[..., 0] - eigenval[..., 2]) ** 2 +
                                (eigenval[..., 1] - eigenval[..., 2]) ** 2,
                                np.sum(eigenval ** 2, axis=-1)))

    return frac_anis


def init_frangi_arrays(image_shape, slice_shape, resize_ratio, save_dir, image_name,
                       z_min=0, z_max=None, lpf_soma_mask=False):
    """
    Initialize the output datasets of the Frangi filtering stage.

    Parameters
    ----------
    image_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    resize_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D resize ratio

    save_dir: str
        saving directory string path

    image_name: str
        name of the input volume image

    z_min: int
        minimum output z-depth in [px]

    z_max: int
        maximum output z-depth in [px]

    lpf_soma_mask: bool
        neuronal body masking flag

    Returns
    -------
    fiber_vec_image: HDF5 dataset (shape=(Z,Y,X,3), dtype=float32)
        initialized fiber orientation volume image

    fiber_vec_colmap: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        initialized orientation colormap image

    frac_anis_image: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized fractional anisotropy image

    frangi_image: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized Frangi-enhanced image

    fiber_mask: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized fiber mask image

    iso_fiber_image: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized fiber image (isotropic resolution)

    neuron_mask: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized neuron mask image

    zsel: NumPy sli object
        selected z-depth range

    tmp_hdf5_lst: list
        list of dictionaries of temporary HDF5 files (file objects and paths),
        where output data chunks are iteratively stored
    """
    # create Frangi volumes subfolder
    save_dir = path.join(save_dir, 'frangi')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # shape copies
    image_shape = image_shape.copy()
    slice_shape = slice_shape.copy()

    # adapt output z-axis shape if required
    if z_min != 0 or z_max is not None:
        if z_max is None:
            z_max = slice_shape[0]
        image_shape[0] = z_max - z_min
    zsel = slice(z_min, z_max, 1)

    # output datasets shape
    image_dims = len(image_shape)
    dset_shape = np.ceil(resize_ratio * image_shape).astype(int)
    slice_shape[0] = dset_shape[0]

    # create list of temporary files
    tmp_hdf5_lst = list()

    # fiber channel
    iso_fiber_path = path.join(save_dir, 'iso_fiber_' + image_name + '.h5')
    iso_fiber_file, iso_fiber_image = create_hdf5_file(iso_fiber_path, dset_shape, slice_shape, dtype='uint8')
    tmp_hdf5_lst.append({'path': iso_fiber_path, 'obj': iso_fiber_file})

    frangi_path = path.join(save_dir, 'frangi_' + image_name + '.h5')
    frangi_file, frangi_image = create_hdf5_file(frangi_path, dset_shape, slice_shape, dtype='uint8')
    tmp_hdf5_lst.append({'path': frangi_path, 'obj': frangi_file})

    fiber_mask_path = path.join(save_dir, 'fiber_msk_' + image_name + '.h5')
    fiber_mask_file, fiber_mask = create_hdf5_file(fiber_mask_path, dset_shape, slice_shape, dtype='uint8')
    tmp_hdf5_lst.append({'path': fiber_mask_path, 'obj': fiber_mask_file})

    frac_anis_path = path.join(save_dir, 'frac_anis_' + image_name + '.h5')
    frac_anis_file, frac_anis_image = create_hdf5_file(frac_anis_path, dset_shape, slice_shape, dtype='uint8')
    tmp_hdf5_lst.append({'path': frac_anis_path, 'obj': frac_anis_file})

    # neuron channel
    if lpf_soma_mask:
        neuron_mask_path = path.join(save_dir, 'neuron_msk_' + image_name + '.h5')
        neuron_mask_file, neuron_mask \
            = create_hdf5_file(neuron_mask_path, dset_shape, slice_shape, dtype='uint8')
        tmp_hdf5_lst.append({'path': neuron_mask_path, 'obj': neuron_mask_file})
    else:
        neuron_mask = None

    # fiber orientation maps
    vec_dset_shape = tuple(list(dset_shape) + [image_dims])
    vec_chunk_shape = tuple(list(slice_shape) + [image_dims])

    fibervec_path = path.join(save_dir, 'fiber_vec_' + image_name + '.h5')
    _, fiber_vec_image = create_hdf5_file(fibervec_path, vec_dset_shape, vec_chunk_shape, dtype='float32')

    orientcol_path = path.join(save_dir, 'fiber_cmap_' + image_name + '.h5')
    orientcol_file, fiber_vec_colmap = create_hdf5_file(orientcol_path, vec_dset_shape, vec_chunk_shape, dtype='uint8')
    tmp_hdf5_lst.append({'path': orientcol_path, 'obj': orientcol_file})

    return fiber_vec_image, fiber_vec_colmap, frac_anis_image, frangi_image, fiber_mask, iso_fiber_image, neuron_mask, \
        zsel, tmp_hdf5_lst


def init_odf_arrays(vec_img_shape, odf_slc_shape, save_dir, odf_degrees=6, odf_scale=15):
    """
    Initialize the output datasets of the ODF analysis stage.

    Parameters
    ----------
    vec_img_shape: numpy.ndarray (shape=(3,), dtype=int)
        vector volume shape [px]

    odf_slc_shape: numpy.ndarray (shape=(3,), dtype=int)
        odf slice shape [px]

    save_dir: str
        saving directory string path

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    odf_scale: int
        fiber ODF resolution (super-voxel side [px])

    Returns
    -------
    odf_img: HDF5 dataset (shape=(X,Y,Z,3), dtype=float32)
        initialized dataset of ODF spherical harmonics coefficients

    bg_mrtrix_img: HDF5 dataset (shape=(X,Y,Z), dtype=uint8)
        initialized background dataset for ODF visualization in Mrtrix3

    odi_pri_img: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized dataset of primary orientation dispersion parameters

    odi_sec_img: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized dataset of secondary orientation dispersion parameters

    odi_tot_img: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized dataset of total orientation dispersion parameters

    odi_anis_img: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        initialized dataset of orientation dispersion anisotropy parameters

    odf_shape: tuple
        ODF image shape

    odi_shape: tuple
        ODI parameter images shape

    odf_tmp_files: list
        list of dictionaries of temporary HDF5 files (file objects and paths)
    """
    # create ODF subfolder
    save_dir = path.join(save_dir, 'odf')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # initialize downsampled background image dataset (HDF5 file)
    bg_shape = np.flip(np.ceil(np.divide(vec_img_shape, odf_scale))).astype(int)
    bg_tmp_path = path.join(save_dir, 'bg_tmp{}.h5'.format(odf_scale))
    bg_tmp_file, bg_mrtrix_img \
        = create_hdf5_file(bg_tmp_path, bg_shape, tuple(np.append(bg_shape[:2], 1)), dtype='uint8')
    bg_tmp_dict = {'path': bg_tmp_path, 'obj': bg_tmp_file}

    # initialize ODF dataset
    num_coeff = get_sph_harm_ncoeff(odf_degrees)
    odf_shape = tuple(list(bg_shape) + [num_coeff])
    odf_tmp_path = path.join(save_dir, 'odf_tmp{}.h5'.format(odf_scale))
    odf_tmp_file, odf_img = create_hdf5_file(odf_tmp_path, odf_shape, tuple(list(np.flip(odf_slc_shape)) + [num_coeff]),
                                             dtype='float32')
    odf_tmp_dict = {'path': odf_tmp_path, 'obj': odf_tmp_file}

    # initialize ODI datasets
    odi_shape = tuple(np.flip(bg_shape))
    odi_pri_tmp_path = path.join(save_dir, 'odi_pri_tmp{}.h5'.format(odf_scale))
    odi_pri_tmp_file, odi_pri_img = create_hdf5_file(odi_pri_tmp_path, odi_shape, odf_slc_shape, dtype='uint8')
    odi_pri_tmp_dict = {'path': odi_pri_tmp_path, 'obj': odi_pri_tmp_file}

    odi_sec_tmp_path = path.join(save_dir, 'odi_sec_tmp{}.h5'.format(odf_scale))
    odi_sec_tmp_file, odi_sec_img = create_hdf5_file(odi_sec_tmp_path, odi_shape, odf_slc_shape, dtype='uint8')
    odi_sec_tmp_dict = {'path': odi_sec_tmp_path, 'obj': odi_sec_tmp_file}

    odi_tot_tmp_path = path.join(save_dir, 'odi_tot_tmp{}.h5'.format(odf_scale))
    odi_tot_tmp_file, odi_tot_img = create_hdf5_file(odi_tot_tmp_path, odi_shape, odf_slc_shape, dtype='uint8')
    odi_tot_tmp_dict = {'path': odi_tot_tmp_path, 'obj': odi_tot_tmp_file}

    odi_anis_tmp_path = path.join(save_dir, 'odi_anis_tmp{}.h5'.format(odf_scale))
    odi_anis_tmp_file, odi_anis_img = create_hdf5_file(odi_anis_tmp_path, odi_shape, odf_slc_shape, dtype='uint8')
    odi_anis_tmp_dict = {'path': odi_anis_tmp_path, 'obj': odi_anis_tmp_file}

    # create list of dictionaries of temporary HDF5 files (object and path)
    odf_tmp_files = [bg_tmp_dict, odf_tmp_dict, odi_pri_tmp_dict, odi_sec_tmp_dict, odi_tot_tmp_dict, odi_anis_tmp_dict]

    return odf_img, bg_mrtrix_img, odi_pri_img, odi_sec_img, odi_tot_img, \
        odi_anis_img, odf_shape, odi_shape, odf_tmp_files


def iterate_frangi_on_slices(image, px_size, px_size_iso, smooth_sigma, save_dir, image_name, max_slice_size=100.0,
                             scales_um=1.25, ch_neuron=0, ch_fiber=1, alpha=0.05, beta=1, gamma=100, dark=False,
                             z_min=0, z_max=None, orient_cmap=False, lpf_soma_mask=False, skeletonize=False,
                             mosaic=False):
    """
    Iteratively apply 3D Frangi filtering to basic TPFM image slices.

    Parameters
    ----------
    image: numpy.ndarray (shape=(Z,Y,X))
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
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

    Returns
    -------
    fiber_vec_image: HDF5 dataset (shape=(Z,Y,X,3), dtype=float32)
        fiber orientation vector image

    fiber_vec_colmap: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap image

    frac_anis_image: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_image: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        Frangi-enhanced volume image (fiber probability volume)

    iso_fiber_image: HDF5 dataset (shape=(Z,Y,X,3), dtype=uint8)
        isotropic fiber image

    fiber_mask: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        fiber mask image

    neuron_mask: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        neuron mask image

    tmp_file_lst: list
        list of temporary file dictionaries
        ('path': file path; 'obj': file object)
    """
    # adjust for graylevel fiber image
    if len(image.shape) == 3:
        ch_fiber = None

    # get info on the input volume image
    image_shape, image_shape_um, image_item_size = get_image_info(image, px_size, mosaic=mosaic)

    # get info on the processed image slices
    in_slice_shape, in_slice_shape_um, out_slice_shape, out_image_shape, px_rsz_ratio, pad = \
        config_frangi_slicing(image_shape, image_item_size, px_size, px_size_iso, smooth_sigma,
                              max_slice_size=max_slice_size)

    # initialize the output volume arrays
    fiber_vec_image, fiber_vec_colmap, frac_anis_image, frangi_image, fiber_mask, \
        iso_fiber_image, neuron_mask, zsel, tmp_file_lst = \
        init_frangi_arrays(image_shape, out_slice_shape, px_rsz_ratio, save_dir, image_name,
                           z_min=z_min, z_max=z_max, lpf_soma_mask=lpf_soma_mask)

    # compute the Frangi filter's scale values in pixel
    scales_px = config_frangi_scales(scales_um, px_size_iso[0])

    # print Frangi filter configuration
    print_frangi_heading(alpha, beta, gamma, scales_um)

    # print iterative analysis information
    print_slicing_info(image_shape_um, in_slice_shape_um, px_size, image_item_size)

    # print neuron masking info
    print_soma_masking(lpf_soma_mask)

    # iteratively apply Frangi filter to basic image slices
    loop_range = np.ceil(np.divide(image_shape, in_slice_shape)).astype(int)
    total_iter = int(np.prod(loop_range))
    with alive_bar(total_iter, title='Image slice', length=33) as bar:
        for z in range(loop_range[0]):

            for y in range(loop_range[1]):

                for x in range(loop_range[2]):

                    # index ranges of the analyzed fiber sli (with padding)
                    rng_in, pad_mat = compute_slice_range(z, y, x, in_slice_shape, image_shape, pad_rng=pad)

                    # output index ranges
                    rng_out, _ = compute_slice_range(z, y, x, out_slice_shape, out_image_shape)

                    # sli fiber image sli
                    fiber_slice = slice_channel(image, rng_in, channel=ch_fiber, mosaic=mosaic)

                    # skip background sli
                    if np.max(fiber_slice) != 0:

                        # preprocess fiber sli
                        iso_fiber_slice = correct_image_anisotropy(fiber_slice, px_rsz_ratio,
                                                                   sigma=smooth_sigma, pad_mat=pad_mat)

                        # crop isotropized fiber sli
                        iso_fiber_slice = crop_slice(iso_fiber_slice, rng_out)

                        # 3D Frangi filter
                        frangi_slice, fiber_vec_slice, eigenval_slice \
                            = frangi_filter(iso_fiber_slice, scales_px=scales_px,
                                            alpha=alpha, beta=beta, gamma=gamma, dark=dark)

                        # generate fractional anisotropy image
                        frac_anis_slice = compute_fractional_anisotropy(eigenval_slice)

                        # generate RGB orientation color map
                        if orient_cmap:
                            orientcol_slice = orient_colormap(fiber_vec_slice)
                        else:
                            orientcol_slice = vector_colormap(fiber_vec_slice)

                        # mask background
                        fiber_vec_slice, orientcol_slice, fiber_mask_slice = \
                            mask_background(frangi_slice, fiber_vec_slice, orientcol_slice,
                                            thresh_method='li', skeletonize=skeletonize, invert_mask=False)

                        # (optional) neuronal body masking
                        if lpf_soma_mask:

                            # neuron sli index ranges (without padding)
                            rng_in, _ = compute_slice_range(z, y, x, in_slice_shape, image_shape)

                            # sli neuron image sli
                            neuron_slice = slice_channel(image, rng_in, channel=ch_neuron, mosaic=mosaic)

                            # resize neuron sli (lateral downsampling)
                            iso_neuron_slice = correct_image_anisotropy(neuron_slice, px_rsz_ratio)

                            # crop isotropized neuron sli
                            iso_neuron_slice = crop_slice(iso_neuron_slice, rng_out)

                            # mask neuronal bodies
                            fiber_vec_slice, orientcol_slice, frac_anis_slice, neuron_mask_slice = \
                                mask_background(iso_neuron_slice, fiber_vec_slice, orientcol_slice, frac_anis_slice,
                                                thresh_method='yen', skeletonize=False, invert_mask=True)

                            # fill neuron mask
                            neuron_mask[rng_out] = (255 * neuron_mask_slice[zsel, ...]).astype(np.uint8)

                        # fill output volumes
                        vec_rng_out = tuple(np.append(rng_out, slice(0, 3, 1)))
                        fiber_vec_image[vec_rng_out] = fiber_vec_slice[zsel, ...]
                        fiber_vec_colmap[vec_rng_out] = orientcol_slice[zsel, ...]
                        iso_fiber_image[rng_out] = iso_fiber_slice[zsel, ...].astype(np.uint8)
                        frac_anis_image[rng_out] = (255 * frac_anis_slice[zsel, ...]).astype(np.uint8)
                        frangi_image[rng_out] = (255 * frangi_slice[zsel, ...]).astype(np.uint8)
                        fiber_mask[rng_out] = (255 * (1 - fiber_mask_slice[zsel, ...])).astype(np.uint8)

                    # advance bar
                    bar()

    return fiber_vec_image, fiber_vec_colmap, frac_anis_image, frangi_image, iso_fiber_image, fiber_mask, neuron_mask, \
        tmp_file_lst


def iterate_odf_on_slices(fiber_vec_dset, iso_fiber_dset, px_size_iso, save_dir, max_slice_size=100.0, tmp_file_lst=[],
                          odf_scale_um=15, odf_degrees=6):
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

    tmp_file_lst: list
        list of dictionaries of temporary HDF5 files (file objects and paths)

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    Returns
    -------
    odf_img: HDF5 dataset (shape=(X,Y,Z,3), dtype=float32)
        dataset of ODF spherical harmonics coefficients

    bg_mrtrix_img: HDF5 dataset (shape=(X,Y,Z), dtype=uint8)
        background dataset for ODF visualization in Mrtrix3

    odi_pri_img: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        dataset of primary orientation dispersion parameters

    odi_sec_img: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        dataset of secondary orientation dispersion parameters

    odi_tot_img: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        dataset of total orientation dispersion parameters

    odi_anis_img: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        dataset of orientation dispersion anisotropy parameters

    tmp_file_lst: list
        updated list of dictionaries of temporary HDF5 files
        (file objects and paths)
    """
    # get info on the input volume of orientation vectors
    vec_img_shape = np.asarray(fiber_vec_dset.shape)[:-1]
    vec_item_size = get_item_bytes(fiber_vec_dset)

    # configure image slicing for ODF analysis
    vec_slc_shape, odf_slc_shape, odf_scale \
        = config_odf_slicing(vec_img_shape, vec_item_size, px_size_iso,
                             odf_scale_um=odf_scale_um, max_slice_size=max_slice_size)

    # print ODF super-voxel size
    print_odf_supervoxel(vec_slc_shape, px_size_iso, odf_scale_um)

    # initialize ODF analysis output volumes
    odf_img, bg_mrtrix_img, odi_pri_img, odi_sec_img, odi_tot_img, odi_anis_img, \
        odf_img_shape, odi_img_shape, odf_tmp_files \
        = init_odf_arrays(vec_img_shape, odf_slc_shape, save_dir, odf_degrees=odf_degrees, odf_scale=odf_scale)
    tmp_file_lst = tmp_file_lst + odf_tmp_files

    # iteratively apply Frangi filter to basic microscopy image slices
    loop_range = np.ceil(np.divide(vec_img_shape, vec_slc_shape)).astype(int)
    total_iter = int(np.prod(loop_range))
    with alive_bar(total_iter, title='Image slice', length=33) as bar:
        for z in range(loop_range[0]):

            for y in range(loop_range[1]):

                for x in range(loop_range[2]):

                    # input index ranges
                    rng_in, _ = compute_slice_range(z, y, x, vec_slc_shape, vec_img_shape)

                    # ODF index ranges
                    rng_odf, _ = compute_slice_range(x, y, z, np.flip(odf_slc_shape), odf_img_shape, flip=True)

                    # ODI index ranges
                    rng_odi, _ = compute_slice_range(z, y, x, odf_slc_shape, odi_img_shape)

                    # load dataset slices to NumPy arrays, transform axes
                    if iso_fiber_dset is None:
                        iso_fiber_slc = None
                    else:
                        iso_fiber_slc = iso_fiber_dset[rng_in]
                    rng_in = tuple(np.append(rng_in, slice(0, 3, 1)))
                    vec_slc = fiber_vec_dset[rng_in]

                    # ODF analysis
                    odf_slc, bg_mrtrix_slc, odi_pri_slc, odi_sec_slc, odi_tot_slc, odi_anis_slc = \
                        compute_scaled_odf(odf_scale, vec_slc, iso_fiber_slc, odf_slc_shape, degrees=odf_degrees)

                    # transform axes
                    odf_slc = transform_axes(odf_slc, swapped=(0, 2), flipped=(1, 2))
                    bg_mrtrix_slc = transform_axes(bg_mrtrix_slc, swapped=(0, 2), flipped=(1, 2))

                    # crop output slices
                    odf_slc = crop_slice(odf_slc, rng_odf, flipped=(0, 1, 2))
                    bg_mrtrix_slc = crop_slice(bg_mrtrix_slc, rng_odf, flipped=(0, 1, 2))
                    odi_pri_slc = crop_slice(odi_pri_slc, rng_odi)
                    odi_sec_slc = crop_slice(odi_sec_slc, rng_odi)
                    odi_tot_slc = crop_slice(odi_tot_slc, rng_odi)
                    odi_anis_slc = crop_slice(odi_anis_slc, rng_odi)

                    # fill datasets
                    bg_mrtrix_img[rng_odf] = bg_mrtrix_slc
                    rng_odf = tuple(np.append(rng_odf, slice(0, odf_slc.shape[-1], 1)))
                    odf_img[rng_odf] = odf_slc
                    odi_pri_img[rng_odi] = (255 * odi_pri_slc).astype(np.uint8)
                    odi_sec_img[rng_odi] = (255 * odi_sec_slc).astype(np.uint8)
                    odi_tot_img[rng_odi] = (255 * odi_tot_slc).astype(np.uint8)
                    odi_anis_img[rng_odi] = (255 * odi_anis_slc).astype(np.uint8)

                    # advance bar
                    bar()

    return odf_img, bg_mrtrix_img, odi_pri_img, odi_sec_img, odi_tot_img, odi_anis_img, tmp_file_lst


def mask_background(image, fiber_vec_slice, orientcol_slice, frac_anis_slice=None,
                    thresh_method='yen', skeletonize=False, invert_mask=False):
    """
    Mask orientation volume arrays.

    Parameters
    ----------
    image: numpy.ndarray (shape=(Z,Y,X))
        fiber (or neuron) fluorescence volume image

    fiber_vec_slice: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        fiber orientation vector sli

    orientcol_slice: numpy.ndarray (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap sli

    frac_anis_slice: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        fractional anisotropy sli

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

    frac_anis_slice: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        fractional anisotropy patch (masked)

    background_mask: numpy.ndarray (shape=(Z,Y,X), dtype=bool)
        background mask
    """
    # generate background mask
    background = create_background_mask(image, thresh_method=thresh_method, skeletonize=skeletonize)

    # invert mask
    if invert_mask:
        background = np.logical_not(background)

    # apply mask to input arrays
    fiber_vec_slice[background, :] = 0
    orientcol_slice[background, :] = 0

    # (optional) mask fractional anisotropy
    if frac_anis_slice is not None:
        frac_anis_slice[background] = 0
        return fiber_vec_slice, orientcol_slice, frac_anis_slice, background

    else:
        return fiber_vec_slice, orientcol_slice, background


def save_frangi_arrays(fiber_vec_colmap, frac_anis_image, frangi_image, fiber_mask, neuron_mask, save_dir, image_name):
    """
    Save the output arrays of the Frangi filter stage to TIF files.

    Parameters
    ----------
    fiber_vec_colmap: HDF5 dataset (shape=(Z,Y,X,3), dtype: uint8)
        orientation colormap image

    frac_anis_image: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_image: HDF5 dataset (shape=(Z,Y,X), dtype: uint8)
        Frangi-enhanced volume image (fiber probability volume)

    fiber_mask: HDF5 dataset (shape=(Z,Y,X), dtype: uint8)
        fiber mask image

    neuron_mask: HDF5 dataset (shape=(Z,Y,X), dtype: uint8)
        neuron mask image

    save_dir: str
        saving directory string path

    image_name: str
        name of the input microscopy volume image

    Returns
    -------
    None
    """
    # final print
    print(color_text(0, 191, 255, "\nSaving Frangi Filter Arrays...\n"))

    # create subfolder
    save_dir = path.join(save_dir, 'frangi')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # save orientation color map to TIF
    save_array('fiber_cmap_' + image_name, save_dir, fiber_vec_colmap)

    # save fractional anisotropy map to TIF
    save_array('frac_anis_' + image_name, save_dir, frac_anis_image)

    # save Frangi-enhanced fiber volume to TIF
    save_array('frangi_' + image_name, save_dir, frangi_image)

    # save masked fiber volume to TIF
    save_array('fiber_msk_' + image_name, save_dir, fiber_mask)

    # save neuron channel volumes to TIF
    if neuron_mask is not None:
        save_array('neuron_msk_' + image_name, save_dir, neuron_mask)


def save_odf_arrays(odf_lst, bg_lst, odi_pri_lst, odi_sec_lst, odi_tot_lst, odi_anis_lst,
                    save_dir, img_name, odf_scales_um):
    """
    Save the output arrays of the ODF analysis stage to TIF and Nifti files.
    Arrays tagged with 'mrtrixview' are preliminarily transformed
    so that ODF maps viewed in Mrtrix3 are spatially consistent
    with the analyzed microscopy volume, and the output TIF files.

    Parameters
    ----------
    odf_lst: list
        list of HDF5 datasets of spherical harmonics coefficients

    bg_mrtrix_lst: list
        list of HDF5 datasets of downsampled background images
        for ODF visualization in Mrtrix3 (fiber channel)

    odi_pri_img_lst: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        list of HDF5 datasets of primary orientation dispersion parameters

    odi_sec_img_lst: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        list of HDF5 datasets of secondary orientation dispersion parameters

    odi_tot_img_lst: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        list of HDF5 datasets of total orientation dispersion parameters

    odi_anis_img_lst: HDF5 dataset (shape=(Z,Y,X), dtype=uint8)
        list of HDF5 datasets of orientation dispersion anisotropy parameters

    save_dir: str
        saving directory string path

    img_name: str
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
    print(color_text(0, 191, 255, "\nSaving ODF Analysis Arrays...\n\n\n"))

    # create ODF subfolder
    save_dir = path.join(save_dir, 'odf')
    if not path.isdir(save_dir):
        mkdir(save_dir)

    # ODF analysis volumes to Nifti files (adjusted view for Mrtrix3)
    for (odf, bg, odi_pri, odi_sec, odi_tot, odi_anis, s) in zip(odf_lst, bg_lst, odi_pri_lst, odi_sec_lst,
                                                                 odi_tot_lst, odi_anis_lst, odf_scales_um):
        save_array(f'bg_mrtrixview_sv{s}_' + img_name, save_dir, bg, format='nii')
        save_array(f'odf_mrtrixview_sv{s}_' + img_name, save_dir, odf, format='nii')
        save_array(f'odi_pri_sv{s}_' + img_name, save_dir, odi_pri, odi=True)
        save_array(f'odi_sec_sv{s}_' + img_name, save_dir, odi_sec, odi=True)
        save_array(f'odi_tot_sv{s}_' + img_name, save_dir, odi_tot, odi=True)
        save_array(f'odi_anis_sv{s}_' + img_name, save_dir, odi_anis, odi=True)
