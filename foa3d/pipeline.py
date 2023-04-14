from os import path
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed

from foa3d.frangi import convert_frangi_scales, frangi_filter
from foa3d.input import get_image_info
from foa3d.odf import (estimate_odf_coeff, generate_odf_background,
                       get_sph_harm_ncoeff, get_sph_harm_norm_factors)
from foa3d.output import save_array
from foa3d.preprocessing import correct_image_anisotropy
from foa3d.printing import (print_analysis_time, print_frangi_info,
                            print_odf_info)
from foa3d.slicing import (config_frangi_batch, config_frangi_slicing,
                           crop_slice, slice_channel)
from foa3d.utils import (create_background_mask, create_memory_map,
                         divide_nonzero, get_available_cores, orient_colormap,
                         vector_colormap)


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
        np.sqrt(0.5 * divide_nonzero(
                (eigenval[..., 0] - eigenval[..., 1]) ** 2 +
                (eigenval[..., 0] - eigenval[..., 2]) ** 2 +
                (eigenval[..., 1] - eigenval[..., 2]) ** 2,
                np.sum(eigenval ** 2, axis=-1)))

    return frac_anis


def init_frangi_volumes(img_shape, slice_shape, resize_ratio, tmp_dir, img_name,
                        z_min=0, z_max=None, lpf_soma_mask=False):
    """
    Initialize the output datasets of the Frangi filtering stage.

    Parameters
    ----------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices analyzed iteratively [px]

    resize_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    tmp_dir: str
        temporary file directory

    img_name: str
        name of the input volume image

    z_min: int
        minimum output z-depth in [px]

    z_max: int
        maximum output z-depth in [px]

    lpf_soma_mask: bool
        neuronal body masking flag

    Returns
    -------
    fiber_vec_img: NumPy memory-map object (shape=(Z,Y,X,3), dtype=float32)
        initialized fiber orientation volume image

    fiber_vec_clr: NumPy memory-map object (shape=(Z,Y,X,3), dtype=uint8)
        initialized orientation colormap image

    frac_anis_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized fractional anisotropy image

    frangi_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized Frangi-enhanced image

    fiber_msk: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized fiber mask image

    iso_fiber_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized fiber image (isotropic resolution)

    neuron_msk: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized neuron mask image

    z_sel: NumPy slice object
        selected z-depth range
    """
    # shape copies
    img_shape = img_shape.copy()
    slice_shape = slice_shape.copy()

    # adapt output z-axis shape if required
    if z_min != 0 or z_max is not None:
        if z_max is None:
            z_max = slice_shape[0]
        img_shape[0] = z_max - z_min
    z_sel = slice(z_min, z_max, 1)

    # output shape
    img_dims = len(img_shape)
    dset_shape = np.ceil(resize_ratio * img_shape).astype(int)
    slice_shape[0] = dset_shape[0]

    # fiber channel memory maps
    iso_fiber_path = path.join(tmp_dir, 'iso_fiber_' + img_name + '.mmap')
    iso_fiber_img = create_memory_map(iso_fiber_path, dset_shape, dtype='uint8')

    frangi_path = path.join(tmp_dir, 'frangi_' + img_name + '.mmap')
    frangi_img = create_memory_map(frangi_path, dset_shape, dtype='uint8')

    fiber_mask_path = path.join(tmp_dir, 'fiber_msk_' + img_name + '.mmap')
    fiber_mask = create_memory_map(fiber_mask_path, dset_shape, dtype='uint8')

    frac_anis_path = path.join(tmp_dir, 'frac_anis_' + img_name + '.mmap')
    frac_anis_img = create_memory_map(frac_anis_path, dset_shape, dtype='uint8')

    # neuron channel memory map
    if lpf_soma_mask:
        neuron_msk_path = path.join(tmp_dir, 'neuron_msk_' + img_name + '.mmap')
        neuron_msk = create_memory_map(neuron_msk_path, dset_shape, dtype='uint8')
    else:
        neuron_msk = None

    # fiber orientation memory maps
    vec_dset_shape = tuple(list(dset_shape) + [img_dims])
    fiber_vec_path = path.join(tmp_dir, 'fiber_vec_' + img_name + '.mmap')
    fiber_vec_img = create_memory_map(fiber_vec_path, vec_dset_shape, dtype='float32')

    fiber_clr_path = path.join(tmp_dir, 'fiber_cmap_' + img_name + '.mmap')
    fiber_vec_clr = create_memory_map(fiber_clr_path, vec_dset_shape, dtype='uint8')

    return fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img, fiber_mask, iso_fiber_img, neuron_msk, z_sel


def init_odf_volumes(vec_img_shape, tmp_dir, odf_scale, odf_degrees=6):
    """
    Initialize the output datasets of the ODF analysis stage.

    Parameters
    ----------
    vec_img_shape: numpy.ndarray (shape=(3,), dtype=int)
        vector volume shape [px]

    odf_slc_shape: numpy.ndarray (shape=(3,), dtype=int)
        odf slice shape [px]

    tmp_dir: str
        temporary file directory

    odf_scale: int
        fiber ODF resolution (super-voxel side [px])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    Returns
    -------
    odf: NumPy memory-map object (shape=(X,Y,Z,3), dtype=float32)
        initialized array of ODF spherical harmonics coefficients

    bg_mrtrix: NumPy memory-map object (shape=(X,Y,Z), dtype=uint8)
        initialized background for ODF visualization in MRtrix3

    odi_pri: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized array of primary orientation dispersion parameters

    odi_sec: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized array of secondary orientation dispersion parameters

    odi_tot: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized array of total orientation dispersion parameters

    odi_anis: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        initialized array of orientation dispersion anisotropy parameters

    vec_tensor_eigen: NumPy memory-map object (shape=(Z,Y,X,3), dtype=uint8)
        initialized array of fiber orientation tensor eigenvalues
    """
    # ODI maps shape
    odi_shape = np.ceil(np.divide(vec_img_shape, odf_scale)).astype(int)

    # create downsampled background memory map
    bg_shape = np.flip(odi_shape)
    bg_tmp_path = path.join(tmp_dir, 'bg_tmp{}.mmap'.format(odf_scale))
    bg_mrtrix = create_memory_map(bg_tmp_path, bg_shape, dtype='uint8')

    # create ODF memory map
    num_coeff = get_sph_harm_ncoeff(odf_degrees)
    odf_shape = tuple(list(odi_shape) + [num_coeff])
    odf_tmp_path = path.join(tmp_dir, 'odf_tmp{}.mmap'.format(odf_scale))
    odf = create_memory_map(odf_tmp_path, odf_shape, dtype='float32')

    # create orientation tensor memory map
    vec_tensor_shape = tuple(list(odi_shape) + [3])
    vec_tensor_tmp_path = path.join(tmp_dir, 'tensor_tmp{}.mmap'.format(odf_scale))
    vec_tensor_eigen = create_memory_map(vec_tensor_tmp_path, vec_tensor_shape, dtype='float32')

    # create ODI memory maps
    odi_pri_tmp_path = path.join(tmp_dir, 'odi_pri_tmp{}.mmap'.format(odf_scale))
    odi_pri = create_memory_map(odi_pri_tmp_path, odi_shape, dtype='uint8')

    odi_sec_tmp_path = path.join(tmp_dir, 'odi_sec_tmp{}.mmap'.format(odf_scale))
    odi_sec = create_memory_map(odi_sec_tmp_path, odi_shape, dtype='uint8')

    odi_tot_tmp_path = path.join(tmp_dir, 'odi_tot_tmp{}.mmap'.format(odf_scale))
    odi_tot = create_memory_map(odi_tot_tmp_path, odi_shape, dtype='uint8')

    odi_anis_tmp_path = path.join(tmp_dir, 'odi_anis_tmp{}.mmap'.format(odf_scale))
    odi_anis = create_memory_map(odi_anis_tmp_path, odi_shape, dtype='uint8')

    return odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, vec_tensor_eigen


def fiber_analysis(img, rng_in, rng_in_neu, rng_out, pad_mat, smooth_sigma, scales_px, px_rsz_ratio, z_sel,
                   fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img, iso_fiber_img, fiber_msk, neuron_msk,
                   ch_neuron=0, ch_fiber=1, alpha=0.05, beta=1, gamma=100,
                   dark=False, orient_cmap=False, lpf_soma_mask=False, mosaic=False):
    """
    Conduct a Frangi-based fiber orientation analysis on basic slices selected from the whole microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        fiber fluorescence volume image

    rng_in: NumPy slice object
        input image range (fibers)

    rng_in_neu: NumPy slice object
        input image range (neurons)

    rng_out: NumPy slice object
        output range

    pad_mat: numpy.ndarray (shape=(Z,Y,X))
        3D image padding range

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (applied to the XY plane)

    scales_px: numpy.ndarray (dtype=int)
        spatial scales [px]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    z_sel: NumPy slice object
        selected z-depth range

    fiber_vec_img: NumPy memory map (shape=(Z,Y,X,3), dtype=float32)
        fiber orientation vector image

    fiber_vec_clr: NumPy memory map (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap image

    frac_anis_img: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        Frangi-enhanced volume image (fiber probability volume)

    iso_fiber_img: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fiber_msk: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        fiber mask image

    neuron_msk: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        neuron mask image

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

    orient_cmap: bool
        if True, generate color maps based on XY-plane orientation angles
        (instead of using the cartesian components of the estimated vectors)

    lpf_soma_mask: bool
        if True, mask neuronal bodies exploiting the autofluorescence
        signal of lipofuscin pigments

    mosaic: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    Returns
    -------
    None
    """
    # slice fiber image slice
    fiber_slice = slice_channel(img, rng_in, channel=ch_fiber, mosaic=mosaic)

    # skip background slice
    if np.max(fiber_slice) != 0:

        # preprocess fiber slice
        iso_fiber_slice, rsz_pad_mat = \
            correct_image_anisotropy(fiber_slice, px_rsz_ratio, sigma=smooth_sigma, pad_mat=pad_mat)

        # 3D Frangi filter
        frangi_slice, fiber_vec_slice, eigenval_slice = \
            frangi_filter(iso_fiber_slice, scales_px=scales_px, alpha=alpha, beta=beta, gamma=gamma, dark=dark)

        # crop resulting slices
        iso_fiber_slice = crop_slice(iso_fiber_slice, rng_out, rsz_pad_mat)
        frangi_slice = crop_slice(frangi_slice, rng_out, rsz_pad_mat)
        fiber_vec_slice = crop_slice(fiber_vec_slice, rng_out, rsz_pad_mat)
        eigenval_slice = crop_slice(eigenval_slice, rng_out, rsz_pad_mat)

        # generate fractional anisotropy image
        frac_anis_slice = compute_fractional_anisotropy(eigenval_slice)

        # generate RGB orientation color map
        if orient_cmap:
            orientcol_slice = orient_colormap(fiber_vec_slice)
        else:
            orientcol_slice = vector_colormap(fiber_vec_slice)

        # mask background
        fiber_vec_slice, orientcol_slice, fiber_mask_slice = \
            mask_background(frangi_slice, fiber_vec_slice, orientcol_slice, thresh_method='li', invert=False)

        # (optional) neuronal body masking
        neuron_mask_slice = None
        if lpf_soma_mask:

            # get neuron image slice
            neuron_slice = slice_channel(img, rng_in_neu, channel=ch_neuron, mosaic=mosaic)

            # resize neuron slice (lateral blurring and downsampling)
            iso_neuron_slice = correct_image_anisotropy(neuron_slice, px_rsz_ratio)

            # crop isotropized neuron slice
            iso_neuron_slice = crop_slice(iso_neuron_slice, rng_out)

            # mask neuronal bodies
            fiber_vec_slice, orientcol_slice, frac_anis_slice, neuron_mask_slice = \
                mask_background(iso_neuron_slice, fiber_vec_slice, orientcol_slice, frac_anis_slice,
                                thresh_method='yen', invert=True)

            # fill memory-mapped output neuron mask
            neuron_msk[rng_out] = (255 * neuron_mask_slice[z_sel, ...]).astype(np.uint8)

        # fill memory-mapped output arrays
        vec_rng_out = tuple(np.append(rng_out, slice(0, 3, 1)))
        fiber_vec_img[vec_rng_out] = fiber_vec_slice[z_sel, ...]
        fiber_vec_clr[vec_rng_out] = orientcol_slice[z_sel, ...]
        iso_fiber_img[rng_out] = iso_fiber_slice[z_sel, ...].astype(np.uint8)
        frac_anis_img[rng_out] = (255 * frac_anis_slice[z_sel, ...]).astype(np.uint8)
        frangi_img[rng_out] = (255 * frangi_slice[z_sel, ...]).astype(np.uint8)
        fiber_msk[rng_out] = (255 * (1 - fiber_mask_slice[z_sel, ...])).astype(np.uint8)


def mask_background(img, fiber_vec_slice, orientcol_slice, frac_anis_slice=None, thresh_method='yen', invert=False):
    """
    Mask orientation volume arrays.

    Parameters
    ----------
    img: numpy.ndarray (shape=(Z,Y,X))
        fiber (or neuron) fluorescence volume image

    fiber_vec_slice: numpy.ndarray (shape=(Z,Y,X,3), dtype=float)
        fiber orientation vector slice

    orientcol_slice: numpy.ndarray (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap slice

    frac_anis_slice: numpy.ndarray (shape=(Z,Y,X), dtype=float)
        fractional anisotropy slice

    thresh_method: str
        thresholding method (refer to skimage.filters)

    invert: bool
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
    background = create_background_mask(img, thresh_method=thresh_method)

    # invert mask
    if invert:
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


def odf_analysis(fiber_vec_img, iso_fiber_img, px_size_iso, save_dir, tmp_dir, img_name, odf_scale_um, odf_norm,
                 odf_degrees=6):
    """
    Estimate 3D fiber ODFs from basic orientation data chunks using parallel threads.

    Parameters
    ----------
    fiber_vec_img: NumPy memory-map object (shape=(Z,Y,X,3), dtype=float32)
        fiber orientation vectors

    iso_fiber_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        isotropic fiber volume

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    save_dir: str
        saving directory string path

    tmp_dir: str
        temporary file directory

    img_name: str
        name of the input volume image

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    odf_norm: numpy.ndarray (dtype: float)
        2D array of spherical harmonics normalization factors

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    Returns
    -------
    None
    """
    # get info on the input volume of orientation vectors
    vec_img_shape = np.asarray(fiber_vec_img.shape)

    # derive the ODF kernel size in [px]
    odf_scale = int(np.ceil(odf_scale_um / px_size_iso[0]))

    # initialize ODF analysis output volumes
    odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, tensor \
        = init_odf_volumes(vec_img_shape[:-1], tmp_dir, odf_scale=odf_scale, odf_degrees=odf_degrees)

    # generate downsampled background for MRtrix3 mrview
    if iso_fiber_img is None:
        bg_mrtrix = generate_odf_background(fiber_vec_img, bg_mrtrix, vxl_side=odf_scale)
    else:
        bg_mrtrix = generate_odf_background(iso_fiber_img, bg_mrtrix, vxl_side=odf_scale)

    # compute ODF coefficients
    odf, odi_pri, odi_sec, odi_tot, odi_anis = \
        estimate_odf_coeff(fiber_vec_img, odf, odi_pri, odi_sec, odi_tot, odi_anis, tensor,
                           vxl_side=odf_scale, odf_norm=odf_norm, odf_degrees=odf_degrees)

    # save memory maps to file
    save_odf_arrays(odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, px_size_iso, save_dir, img_name, odf_scale_um)


def parallel_frangi_on_slices(img, px_size, px_size_iso, smooth_sigma, save_dir, tmp_dir, img_name, frangi_sigma_um,
                              ch_neuron=0, ch_fiber=1, alpha=0.05, beta=1, gamma=100, dark=False, z_min=0, z_max=None,
                              orient_cmap=False, lpf_soma_mask=False, mosaic=False,
                              max_ram_mb=None, jobs_to_cores=0.8, backend='threading'):
    """
    Apply 3D Frangi filtering to basic TPFM image slices using parallel threads.

    Parameters
    ----------
    img: NumPy memory-map object (shape=(Z,Y,X))
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

    tmp_dir: str
        temporary file directory

    img_name: str
        name of the input volume image

    frangi_sigma_um: list (dtype=float)
        Frangi filter scales in [μm]

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

    mosaic: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    max_ram_mb: float
        maximum RAM available to the Frangi filtering stage [MB]

    jobs_to_cores: float
        max number of jobs relative to the available CPU cores
        (default: 80%)

    backend: str
        backend module employed by joblib.Parallel

    Returns
    -------
    fiber_vec_img: NumPy memory map (shape=(Z,Y,X,3), dtype=float32)
        fiber orientation vector image

    fiber_vec_clr: NumPy memory map (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap image

    frac_anis_img: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        Frangi-enhanced volume image (fiber probability volume)

    iso_fiber_img: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fiber_msk: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        fiber mask image

    neuron_msk: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        neuron mask image
    """
    # adjust for graylevel fiber image
    if len(img.shape) == 3:
        ch_fiber = None

    # get info on the input volume image
    img_shape, img_shape_um, img_item_size, ch_fiber = \
        get_image_info(img, px_size, ch_fiber, mosaic=mosaic)

    # configure batch of basic image slices analyzed in parallel
    batch_size, max_slice_size = \
        config_frangi_batch(frangi_sigma_um, max_ram_mb=max_ram_mb, jobs_to_cores=jobs_to_cores)

    # convert the spatial scales of the Frangi filter to pixel
    frangi_sigma_px = convert_frangi_scales(frangi_sigma_um, px_size_iso[0])

    # get info on the processed image slices
    rng_in_lst, rng_in_neu_lst, rng_out_lst, pad_mat_lst, \
        in_slice_shape_um, out_slice_shape, px_rsz_ratio, tot_slice_num, batch_size = \
        config_frangi_slicing(img_shape, img_item_size, px_size, px_size_iso,
                              smooth_sigma, frangi_sigma_px, lpf_soma_mask, batch_size, slice_size=max_slice_size)

    # initialize output arrays
    fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img, fiber_msk, iso_fiber_img, neuron_msk, z_sel = \
        init_frangi_volumes(img_shape, out_slice_shape, px_rsz_ratio, tmp_dir, img_name,
                            z_min=z_min, z_max=z_max, lpf_soma_mask=lpf_soma_mask)

    # print Frangi filter configuration
    print_frangi_info(alpha, beta, gamma, frangi_sigma_um, img_shape_um, in_slice_shape_um, tot_slice_num,
                      px_size, img_item_size, lpf_soma_mask)

    # parallel Frangi filter-based fiber orientation analysis of microscopy sub-volumes
    start_time = perf_counter()
    with Parallel(n_jobs=batch_size, backend=backend, verbose=100, max_nbytes=None) as parallel:
        parallel(
            delayed(fiber_analysis)(img,
                                    rng_in_lst[i], rng_in_neu_lst[i], rng_out_lst[i], pad_mat_lst[i],
                                    smooth_sigma, frangi_sigma_px, px_rsz_ratio, z_sel,
                                    fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img,
                                    iso_fiber_img, fiber_msk, neuron_msk,
                                    ch_neuron=ch_neuron, ch_fiber=ch_fiber, alpha=alpha, beta=beta, gamma=gamma,
                                    dark=dark, orient_cmap=orient_cmap, lpf_soma_mask=lpf_soma_mask, mosaic=mosaic)
            for i in range(tot_slice_num))

    # save Frangi output arrays
    save_frangi_arrays(fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img, fiber_msk, neuron_msk,
                       px_size_iso, save_dir, img_name)

    # print Frangi filtering time
    print_analysis_time(start_time)

    return fiber_vec_img, iso_fiber_img


def parallel_odf_on_scales(fiber_vec_img, iso_fiber_img, px_size_iso, save_dir, tmp_dir, img_name,
                           odf_scales_um, odf_degrees=6, backend='loky'):
    """
    Iterate over the required spatial scales and apply the parallel ODF analysis
    implemented in parallel_odf_on_slices().

    Parameters
    ----------
    fiber_vec_img: NumPy memory-map object (shape=(Z,Y,X,3), dtype=float32)
        fiber orientation vector image

    iso_fiber_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    save_dir: str
        saving directory string path

    tmp_dir: str
        temporary file directory

    img_name: str
        name of the input volume image

    odf_scales_um: list (dtype: float)
        list of fiber ODF resolution values (super-voxel sides [μm])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    backend: str
        backend module employed by joblib.Parallel

    Returns
    -------
    None
    """
    # get ODF analysis start time
    start_time = perf_counter()

    # print ODF analysis heading
    print_odf_info(odf_scales_um, odf_degrees)

    # compute spherical harmonics normalization factors (once for all scales)
    norm_factors = get_sph_harm_norm_factors(odf_degrees)

    # number of logical cores
    num_cpu = get_available_cores()

    # parallel ODF analysis of fiber orientation vectors
    # over the required spatial scales
    n_jobs = min(num_cpu, len(odf_scales_um))
    with Parallel(n_jobs=n_jobs, backend=backend, verbose=100, max_nbytes=None) as parallel:
        parallel(delayed(odf_analysis)(fiber_vec_img, iso_fiber_img, px_size_iso, save_dir, tmp_dir, img_name,
                                       odf_norm=norm_factors, odf_degrees=odf_degrees, odf_scale_um=s)
                 for s in odf_scales_um)

    # print ODF analysis time
    print_analysis_time(start_time)


def save_frangi_arrays(fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img,
                       fiber_msk, neuron_msk, px_size, save_dir, img_name):
    """
    Save the output arrays of the Frangi filter stage to TIF files.

    Parameters
    ----------
    fiber_vec_img: NumPy memory map (shape=(Z,Y,X,3), dtype=float32)
        fiber orientation vector image

    fiber_vec_clr: NumPy memory map (shape=(Z,Y,X,3), dtype=uint8)
        orientation colormap image

    frac_anis_img: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        Frangi-enhanced volume image (fiber probability)

    fiber_msk: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        fiber mask image

    neuron_msk: NumPy memory map (shape=(Z,Y,X), dtype=uint8)
        neuron mask image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size (Z,Y,X) [μm]

    save_dir: str
        saving directory string path

    img_name: str
        name of the input microscopy volume image

    Returns
    -------
    None
    """
    # save orientation color map to TIF
    save_array('fiber_vec_' + img_name, save_dir, fiber_vec_img, format='npy')

    # save orientation color map to TIF
    save_array('fiber_cmap_' + img_name, save_dir, fiber_vec_clr, px_size)

    # save fractional anisotropy map to TIF
    save_array('frac_anis_' + img_name, save_dir, frac_anis_img, px_size)

    # save Frangi-enhanced fiber volume to TIF
    save_array('frangi_' + img_name, save_dir, frangi_img, px_size)

    # save masked fiber volume to TIF
    save_array('fiber_msk_' + img_name, save_dir, fiber_msk, px_size)

    # save masked neuron volume to TIF
    if neuron_msk is not None:
        save_array('neuron_msk_' + img_name, save_dir, neuron_msk, px_size)


def save_odf_arrays(odf, bg, odi_pri, odi_sec, odi_tot, odi_anis, px_size, save_dir, img_name, odf_scale_um):
    """
    Save the output arrays of the ODF analysis stage to TIF and Nifti files.
    Arrays tagged with 'mrtrixview' are preliminarily transformed
    so that ODF maps viewed in MRtrix3 are spatially consistent
    with the analyzed microscopy volume, and the output TIF files.

    Parameters
    ----------
    odf_img: NumPy memory-map object (shape=(X,Y,Z,3), dtype=float32)
        ODF spherical harmonics coefficients

    bg_mrtrix_img: NumPy memory-map object (shape=(X,Y,Z), dtype=uint8)
        background for ODF visualization in MRtrix3

    odi_pri_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        primary orientation dispersion parameter

    odi_sec_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        secondary orientation dispersion parameter

    odi_tot_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        total orientation dispersion parameter

    odi_anis_img: NumPy memory-map object (shape=(Z,Y,X), dtype=uint8)
        orientation dispersion anisotropy parameter

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size (Z,Y,X) [μm]

    save_dir: str
        saving directory string path

    img_name: str
        name of the input volume image

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    Returns
    -------
    None
    """
    # ODF analysis volumes to Nifti files (adjusted view for MRtrix3)
    save_array(f'bg_mrtrixview_sv{odf_scale_um}_' + img_name, save_dir, bg, format='nii')
    save_array(f'odf_mrtrixview_sv{odf_scale_um}_' + img_name, save_dir, odf, format='nii')
    save_array(f'odi_pri_sv{odf_scale_um}_' + img_name, save_dir, odi_pri, px_size, odi=True)
    save_array(f'odi_sec_sv{odf_scale_um}_' + img_name, save_dir, odi_sec, px_size, odi=True)
    save_array(f'odi_tot_sv{odf_scale_um}_' + img_name, save_dir, odi_tot, px_size, odi=True)
    save_array(f'odi_anis_sv{odf_scale_um}_' + img_name, save_dir, odi_anis, px_size, odi=True)
