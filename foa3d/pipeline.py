import shutil
from os import path
from time import perf_counter

import numpy as np
import psutil
from joblib import Parallel, delayed

from foa3d.frangi import frangi_filter
from foa3d.input import get_image_info, get_frangi_config
from foa3d.odf import (compute_odf_map, generate_odf_background,
                       get_sph_harm_ncoeff, get_sph_harm_norm_factors)
from foa3d.output import save_array
from foa3d.preprocessing import correct_image_anisotropy
from foa3d.printing import (print_analysis_time, print_frangi_info,
                            print_odf_info)
from foa3d.slicing import (config_frangi_batch, config_frangi_slicing,
                           crop_slice, crop_slice_lst, slice_channel)
from foa3d.utils import (create_background_mask, create_hdf5_dset,
                         create_memory_map, divide_nonzero,
                         get_available_cores, get_item_size, hsv_orient_cmap,
                         rgb_orient_cmap)


def compute_fractional_anisotropy(eigenval):
    """
    Compute structure tensor fractional anisotropy
    as in Schilling et al. (2018).

    Parameters
    ----------
    eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        structure tensor eigenvalues (best local spatial scale)

    Returns
    -------
    frac_anis: numpy.ndarray (shape=(3,), dtype=float)
        fractional anisotropy
    """

    frac_anis = \
        np.sqrt(0.5 * divide_nonzero(
                np.square((eigenval[..., 0] - eigenval[..., 1])) +
                np.square((eigenval[..., 0] - eigenval[..., 2])) +
                np.square((eigenval[..., 1] - eigenval[..., 2])),
                np.sum(eigenval ** 2, axis=-1)))

    return frac_anis


def init_frangi_volumes(img_shape, slice_shape, resize_ratio, tmp_dir, z_rng=(0, None), mask_lpf=False, ram=None):
    """
    Initialize the output datasets of the Frangi filtering stage.

    Parameters
    ----------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    slice_shape: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    resize_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    tmp_dir: str
        temporary file directory

    z_rng: int
        output z-range in [px]

    mask_lpf: bool
        if True, mask neuronal bodies exploiting the autofluorescence
        signal of lipofuscin pigments

    ram: float
        maximum RAM available to the Frangi filtering stage [B]

    Returns
    -------
    fiber_dset_path: str
        path to initialized fiber orientation HDF5 dataset
        (not fitting the available RAM)

    fiber_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        initialized fiber orientation volume image

    fiber_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        initialized orientation colormap image

    frac_anis_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized Frangi-enhanced image

    fiber_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized fiber mask image

    iso_fiber_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized fiber image (isotropic resolution)

    soma_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized soma mask image

    z_sel: NumPy slice object
        selected z-depth range
    """

    # shape copies
    img_shape = img_shape.copy()
    slice_shape = slice_shape.copy()

    # adapt output z-axis shape if required
    z_min, z_max = z_rng
    if z_min != 0 or z_max is not None:
        if z_max is None:
            z_max = slice_shape[0]
        img_shape[0] = z_max - z_min
    z_sel = slice(z_min, z_max, 1)

    # output shape
    img_dims = len(img_shape)
    tot_shape = tuple(np.ceil(resize_ratio * img_shape).astype(int))
    slice_shape[0] = tot_shape[0]

    # fiber channel arrays
    iso_fiber_img, _ = init_volume(tot_shape, dtype='uint8', chunks=slice_shape, name='iso_fiber', tmp=tmp_dir, ram=ram)
    frangi_img, _ = init_volume(tot_shape, dtype='uint8', chunks=slice_shape, name='frangi', tmp=tmp_dir, ram=ram)
    fiber_msk, _ = init_volume(tot_shape, dtype='uint8', chunks=slice_shape, name='fiber_msk', tmp=tmp_dir, ram=ram)
    frac_anis_img, _ = init_volume(tot_shape, dtype='uint8', chunks=slice_shape, name='frac_anis', tmp=tmp_dir, ram=ram)

    # soma channel array
    if mask_lpf:
        soma_msk, _ = init_volume(tot_shape, dtype='uint8', chunks=slice_shape, name='soma_msk', tmp=tmp_dir, ram=ram)
    else:
        soma_msk = None

    # fiber orientation arrays
    vec_shape = tot_shape + (img_dims,)
    vec_slice_shape = tuple(slice_shape) + (img_dims,)
    fiber_vec_img, fiber_dset_path = \
        init_volume(vec_shape, dtype='float32', chunks=vec_slice_shape, name='fiber_vec', tmp=tmp_dir, ram=ram)
    fiber_vec_clr, _ = \
        init_volume(vec_shape, dtype='uint8', chunks=vec_slice_shape, name='fiber_cmap', tmp=tmp_dir, ram=ram)

    return fiber_dset_path, fiber_vec_img, fiber_vec_clr, frac_anis_img, \
        frangi_img, fiber_msk, iso_fiber_img, soma_msk, z_sel


def init_odf_volumes(vec_img_shape, tmp_dir, odf_scale, odf_degrees=6, ram=None):
    """
    Initialize the output datasets of the ODF analysis stage.

    Parameters
    ----------
    vec_img_shape: tuple
        vector volume shape [px]

    tmp_dir: str
        temporary file directory

    odf_scale: int
        fiber ODF resolution (super-voxel side [px])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    ram: float
        maximum RAM available to the ODF generation stage [B]

    Returns
    -------
    odf: NumPy memory-map object or HDF5 dataset (axis order=(X,Y,Z,C), dtype=float32)
        initialized array of ODF spherical harmonics coefficients

    bg_mrtrix: NumPy memory-map object or HDF5 dataset (axis order=(X,Y,Z), dtype=uint8)
        initialized background for ODF visualization in MRtrix3

    odi_pri: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized array of primary orientation dispersion parameters

    odi_sec: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized array of secondary orientation dispersion parameters

    odi_tot: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized array of total orientation dispersion parameters

    odi_anis: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized array of orientation dispersion anisotropy parameters

    vec_tensor_eigen: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        initialized array of fiber orientation tensor eigenvalues
    """

    # ODI maps shape
    odi_shape = tuple(np.ceil(np.divide(vec_img_shape, odf_scale)).astype(int))

    # create downsampled background memory map
    bg_shape = tuple(np.flip(odi_shape))
    bg_mrtrix, _ = init_volume(bg_shape, dtype='uint8', chunks=tuple(bg_shape[:2]) + (1,),
                               name='bg_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)

    # create ODF memory map
    num_coeff = get_sph_harm_ncoeff(odf_degrees)
    odf_shape = odi_shape + (num_coeff,)
    odf, _ = init_volume(odf_shape, dtype='float32', chunks=(1, 1, 1, num_coeff),
                         name='odf_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)

    # create orientation tensor memory map
    vec_tensor_shape = odi_shape + (3,)
    vec_tensor_eigen, _ = \
        init_volume(vec_tensor_shape, dtype='float32', chunks=(1, 1, 1, 3),
                    name='tensor_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)

    # create ODI memory maps
    odi_pri, _ = init_volume(odi_shape, dtype='uint8',
                             name='odi_pri_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)
    odi_sec, _ = init_volume(odi_shape, dtype='uint8',
                             name='odi_sec_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)
    odi_tot, _ = init_volume(odi_shape, dtype='uint8',
                             name='odi_tot_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)
    odi_anis, _ = init_volume(odi_shape, dtype='uint8',
                              name='odi_anis_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)

    return odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, vec_tensor_eigen


def init_volume(shape, dtype, chunks=True, name='tmp', tmp=None, mmap_mode='r+', ram=None):
    """
    Initialize output volume as an empty HDF5 dataset
    or a memory-mapped array depending on available RAM.

    Parameters
    ----------
    shape: tuple (dtype=int)
        data shape

    dtype: str
        data type

    chunks: tuple (dtype=int) or bool
        shape of the chunked storage layout (HDF5 only, default: auto chunking)

    name: str
        optional temporary filename

    tmp: str
        temporary file directory

    mmap_mode: str
        file opening mode (memory-mapped object only)

    ram: float
        maximum RAM available to the Frangi filtering stage [B]

    Returns
    -------
    vol: NumPy memory-map object or HDF5 dataset
        initialized data volume

    hdf5_path: str
        path to the HDF5 file
    """

    # get maximum RAM and initialized array memory size
    if ram is None:
        ram = psutil.virtual_memory()[1]
    item_sz = get_item_size(dtype)
    vol_sz = item_sz * np.prod(shape)

    # create memory-mapped array or HDF5 file depending on available memory resources
    if vol_sz >= ram:
        vol, hdf5_path = create_hdf5_dset(shape, dtype, chunks=chunks, name=name, tmp=tmp)
    else:
        vol = create_memory_map(shape, dtype, name=name, tmp_dir=tmp, mmap_mode=mmap_mode)
        hdf5_path = None

    return vol, hdf5_path


def fiber_analysis(img, rng_in, rng_in_neu, rng_out, pad, ovlp, smooth_sigma, scales_px, px_rsz_ratio, z_sel,
                   fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img, iso_fiber_img, fiber_msk, soma_msk,
                   ch_lpf=0, ch_mye=1, alpha=0.05, beta=1, gamma=100, dark=False, mask_lpf=False, hsv_vec_cmap=False,
                   pad_mode='reflect', is_tiled=False, fiber_thresh='li', soma_thresh='yen'):
    """
    Conduct a Frangi-based fiber orientation analysis on basic slices selected from the whole microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X))
        fiber fluorescence volume image

    rng_in: NumPy slice object
        input image range (fibers)

    rng_in_neu: NumPy slice object
        input image range (neurons)

    rng_out: NumPy slice object
        output range

    pad: numpy.ndarray (axis order=(Z,Y,X))
        image padding array

    ovlp: numpy.ndarray (shape=(3,), dtype=int)
        slice overlap range
        along each axis side

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (applied to the XY plane)

    scales_px: numpy.ndarray (dtype=int)
        spatial scales [px]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    z_sel: NumPy slice object
        selected z-depth range

    fiber_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    fiber_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    frac_anis_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced volume image (fiber probability volume)

    iso_fiber_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fiber_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    soma_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        soma mask image

    ch_lpf: int
        neuronal bodies channel

    ch_mye: int
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

    hsv_vec_cmap: bool
        if True, generate a HSV orientation color map based on XY-plane orientation angles
        (instead of an RGB map using the cartesian components of the estimated vectors)

    mask_lpf: bool
        if True, mask neuronal bodies exploiting the autofluorescence
        signal of lipofuscin pigments

    pad_mode: str
        image padding mode

    is_tiled: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    fiber_thresh: str
        enhanced fiber channel thresholding method

    soma_thresh: str
        soma channel thresholding method

    Returns
    -------
    None
    """

    # slice fiber image slice
    fiber_slice = slice_channel(img, rng_in, channel=ch_mye, is_tiled=is_tiled)

    # skip background slice
    if np.max(fiber_slice) != 0:

        # preprocess fiber slice
        iso_fiber_slice, rsz_pad = \
            correct_image_anisotropy(fiber_slice, px_rsz_ratio, sigma=smooth_sigma, pad=pad)

        # pad fiber slice if required
        if rsz_pad is not None:
            iso_fiber_slice = np.pad(iso_fiber_slice, rsz_pad, mode=pad_mode)

        # 3D Frangi filter
        frangi_slice, fiber_vec_slice, eigenval_slice = \
            frangi_filter(iso_fiber_slice, scales_px=scales_px, alpha=alpha, beta=beta, gamma=gamma, dark=dark)

        # crop resulting slices
        iso_fiber_slice, frangi_slice, fiber_vec_slice, eigenval_slice = \
            crop_slice_lst([iso_fiber_slice, frangi_slice, fiber_vec_slice, eigenval_slice], rng_out, ovlp=ovlp)

        # generate fractional anisotropy image
        frac_anis_slice = compute_fractional_anisotropy(eigenval_slice)

        # generate fiber orientation color map
        fiber_clr_slice = hsv_orient_cmap(fiber_vec_slice) if hsv_vec_cmap else rgb_orient_cmap(fiber_vec_slice)

        # remove background
        fiber_vec_slice, fiber_clr_slice, fiber_msk_slice = \
            mask_background(frangi_slice, fiber_vec_slice, fiber_clr_slice, method=fiber_thresh, invert=False)

        # (optional) neuronal body masking
        if mask_lpf:

            # get soma image slice
            soma_slice = slice_channel(img, rng_in_neu, channel=ch_lpf, is_tiled=is_tiled)

            # resize soma slice (lateral blurring and downsampling)
            iso_soma_slice, _ = correct_image_anisotropy(soma_slice, px_rsz_ratio)

            # crop isotropized soma slice
            iso_soma_slice = crop_slice(iso_soma_slice, rng_out)

            # mask neuronal bodies
            fiber_vec_slice, fiber_clr_slice, frac_anis_slice, soma_msk_slice = \
                mask_background(iso_soma_slice, fiber_vec_slice, fiber_clr_slice, frac_anis_slice,
                                method=soma_thresh, invert=True)
        else:
            soma_msk_slice = None

        # fill memory-mapped output arrays
        fill_frangi_volumes(fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img, iso_fiber_img, fiber_msk, soma_msk,
                            fiber_vec_slice, fiber_clr_slice, frac_anis_slice, frangi_slice, iso_fiber_slice,
                            fiber_msk_slice, soma_msk_slice, rng_out, z_sel)


def fill_frangi_volumes(fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img, iso_fiber_img, fiber_msk, soma_msk,
                        fiber_vec_slice, fiber_clr_slice, frac_anis_slice, frangi_slice, iso_fiber_slice,
                        fiber_msk_slice, soma_msk_slice, rng_out, z_sel):
    """
    Fill the memory-mapped output arrays of the Frangi filtering stage.

    Parameters
    ----------
    fiber_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    fiber_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    frac_anis_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced image (fiber probability image)

    iso_fiber_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fiber_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    soma_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        soma mask image

    fiber_vec_slice: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        fiber orientation vector image slice

    fiber_clr_slice: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        orientation colormap image slice

    frac_anis_slice: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        fractional anisotropy image slice

    frangi_slice: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        Frangi-enhanced image slice

    iso_fiber_slice: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image slice

    fiber_msk_slice: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image slice

    soma_msk_slice: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        soma mask image slice

    rng_out: tuple
        3D slice output index range

    z_sel: NumPy slice object
        selected z-depth range

    Returns
    -------
    None
    """

    # fill memory-mapped output arrays
    vec_rng_out = tuple(np.append(rng_out, slice(0, 3, 1)))
    fiber_vec_img[vec_rng_out] = fiber_vec_slice[z_sel, ...]
    fiber_vec_clr[vec_rng_out] = fiber_clr_slice[z_sel, ...]
    iso_fiber_img[rng_out] = iso_fiber_slice[z_sel, ...].astype(np.uint8)
    frac_anis_img[rng_out] = (255 * frac_anis_slice[z_sel, ...]).astype(np.uint8)
    frangi_img[rng_out] = (255 * frangi_slice[z_sel, ...]).astype(np.uint8)
    fiber_msk[rng_out] = (255 * (1 - fiber_msk_slice[z_sel, ...])).astype(np.uint8)

    # fill memory-mapped output soma mask, if available
    if soma_msk is not None:
        soma_msk[rng_out] = (255 * soma_msk_slice[z_sel, ...]).astype(np.uint8)


def mask_background(img, fiber_vec_slice, fiber_clr_slice, frac_anis_slice=None, method='yen', invert=False):
    """
    Mask orientation volume arrays.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        fiber (or neuron) fluorescence volume image

    fiber_vec_slice: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        fiber orientation vector slice

    fiber_clr_slice: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        fiber orientation colormap slice

    frac_anis_slice: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        fractional anisotropy slice

    method: str
        thresholding method (refer to skimage.filters)

    invert: bool
        mask inversion flag

    Returns
    -------
    fiber_vec_slice: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        orientation vector patch (masked)

    orientcol_slice: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap patch (masked)

    frac_anis_slice: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        fractional anisotropy patch (masked)

    background_mask: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
        background mask
    """

    # generate background mask
    background = create_background_mask(img, method=method)

    # invert mask
    if invert:
        background = np.logical_not(background)

    # apply mask to input arrays
    fiber_vec_slice[background, :] = 0
    fiber_clr_slice[background, :] = 0

    # (optional) mask fractional anisotropy
    if frac_anis_slice is not None:
        frac_anis_slice[background] = 0
        return fiber_vec_slice, fiber_clr_slice, frac_anis_slice, background

    else:
        return fiber_vec_slice, fiber_clr_slice, background


def odf_analysis(fiber_vec_img, iso_fiber_img, px_size_iso, save_dir, tmp_dir, img_name, odf_scale_um,
                 odf_norm, odf_deg=6, ram=None):
    """
    Estimate 3D fiber ODFs from basic orientation data chunks using parallel threads.

    Parameters
    ----------
    fiber_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vectors

    iso_fiber_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
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

    odf_deg: int
        degrees of the spherical harmonics series expansion

    ram: float
        maximum RAM available to the ODF analysis stage [B]

    Returns
    -------
    None
    """

    # derive the ODF kernel size in [px]
    odf_scale = int(np.ceil(odf_scale_um / px_size_iso[0]))

    # initialize ODF analysis output volumes
    odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, tensor \
        = init_odf_volumes(fiber_vec_img.shape[:-1], tmp_dir, odf_scale=odf_scale, odf_degrees=odf_deg, ram=ram)

    # generate downsampled background for MRtrix3 mrview
    bg_img = fiber_vec_img if iso_fiber_img is None else iso_fiber_img
    generate_odf_background(bg_img, bg_mrtrix, vxl_side=odf_scale)

    # compute ODF coefficients
    odf = compute_odf_map(fiber_vec_img, odf, odi_pri, odi_sec, odi_tot, odi_anis, tensor, odf_scale, odf_norm,
                          odf_deg=odf_deg)

    # save memory maps to file
    save_odf_arrays(odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, px_size_iso, save_dir, img_name, odf_scale_um)


def parallel_frangi_on_slices(img, cli_args, save_dir, tmp_dir, img_name,
                              ram=None, jobs=4, backend='threading', is_tiled=False, verbose=100):
    """
    Apply 3D Frangi filtering to basic TPFM image slices using parallel threads.

    Parameters
    ----------
    img: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X))
        microscopy volume image

    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    save_dir: str
        saving directory string path

    tmp_dir: str
        temporary file directory

    img_name: str
        name of the microscopy volume image

    ram: float
        maximum RAM available to the Frangi filtering stage [B]

    jobs: int
        number of parallel jobs (threads)
        used by the Frangi filtering stage

    backend: str
        backend module employed by joblib.Parallel

    is_tiled: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    verbose: int
        joblib verbosity level

    Returns
    -------
    fiber_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    fiber_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    frac_anis_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced volume image (fiber probability volume)

    iso_fiber_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fiber_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    soma_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        soma mask image
    """

    # get Frangi filter configuration
    alpha, beta, gamma, frangi_sigma_px, frangi_sigma_um, smooth_sigma, px_size, px_size_iso, \
        z_rng, ch_lpf, ch_mye, mask_lpf, hsv_vec_cmap, img_name = get_frangi_config(cli_args, img_name)

    # get info on the input volume image
    img_shape, img_shape_um, img_item_size, ch_mye, mask_lpf = \
        get_image_info(img, px_size, mask_lpf, ch_mye, is_tiled=is_tiled)

    # configure batch of basic image slices analyzed in parallel
    batch_size, max_slice_size = \
        config_frangi_batch(frangi_sigma_um, ram=ram, jobs=jobs)

    # get info on the processed image slices
    rng_in_lst, rng_in_lpf_lst, rng_out_lst, pad_lst, \
        in_slice_shape_um, out_slice_shape, px_rsz_ratio, rsz_ovlp, tot_slice_num, batch_size = \
        config_frangi_slicing(img_shape, img_item_size, px_size, px_size_iso,
                              smooth_sigma, frangi_sigma_um, mask_lpf, batch_size, slice_size=max_slice_size)

    # initialize output arrays
    fiber_dset_path, fiber_vec_img, fiber_vec_clr, frac_anis_img, \
        frangi_img, fiber_msk, iso_fiber_img, soma_msk, z_sel = \
        init_frangi_volumes(img_shape, out_slice_shape, px_rsz_ratio, tmp_dir,
                            z_rng=z_rng, mask_lpf=mask_lpf, ram=ram)

    # print Frangi filter configuration
    print_frangi_info(alpha, beta, gamma, frangi_sigma_um, img_shape_um, in_slice_shape_um, tot_slice_num,
                      px_size, img_item_size, mask_lpf)

    # parallel Frangi filter-based fiber orientation analysis of microscopy sub-volumes
    start_time = perf_counter()
    with Parallel(n_jobs=batch_size, backend=backend, verbose=verbose, max_nbytes=None) as parallel:
        parallel(
            delayed(fiber_analysis)(img,
                                    rng_in_lst[i], rng_in_lpf_lst[i], rng_out_lst[i], pad_lst[i], rsz_ovlp,
                                    smooth_sigma, frangi_sigma_px, px_rsz_ratio, z_sel,
                                    fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img,
                                    iso_fiber_img, fiber_msk, soma_msk, ch_lpf=ch_lpf, ch_mye=ch_mye,
                                    alpha=alpha, beta=beta, gamma=gamma, mask_lpf=mask_lpf,
                                    hsv_vec_cmap=hsv_vec_cmap, is_tiled=is_tiled)
            for i in range(tot_slice_num))

    # save Frangi output arrays
    save_frangi_arrays(fiber_dset_path, fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img, fiber_msk, soma_msk,
                       px_size_iso, save_dir, img_name)

    # print Frangi filtering time
    print_analysis_time(start_time)

    return fiber_vec_img, iso_fiber_img, px_size_iso, img_name


def parallel_odf_at_scales(fiber_vec_img, iso_fiber_img, cli_args, px_size_iso, save_dir, tmp_dir, img_name,
                           backend='loky', ram=None, verbose=100):
    """
    Iterate over the required spatial scales and apply the parallel ODF analysis
    implemented in parallel_odf_on_slices().

    Parameters
    ----------
    fiber_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    iso_fiber_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    px_size_iso: numpy.ndarray (axis order=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    save_dir: str
        saving directory string path

    tmp_dir: str
        temporary file directory

    img_name: str
        name of the input volume image

    backend: str
        backend module employed by joblib.Parallel

    ram: float
        maximum RAM available to the ODF analysis stage [B]

    verbose: int
        joblib verbosity level

    Returns
    -------
    None
    """

    # get ODF analysis start time
    start_time = perf_counter()

    # print ODF analysis heading
    print_odf_info(cli_args.odf_res, cli_args.odf_deg)

    # compute spherical harmonics normalization factors (once for all scales)
    odf_norm = get_sph_harm_norm_factors(cli_args.odf_deg)

    # get number of logical cores
    num_cpu = get_available_cores()

    # generate pixel size if not provided
    if px_size_iso is None:
        px_size_iso = cli_args.px_size_z * np.ones((3,))

    # parallel ODF analysis of fiber orientation vectors
    # over the required spatial scales
    n_jobs = min(num_cpu, len(cli_args.odf_res))
    with Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose, max_nbytes=None) as parallel:
        parallel(delayed(odf_analysis)(fiber_vec_img, iso_fiber_img, px_size_iso, save_dir, tmp_dir, img_name,
                                       odf_norm=odf_norm, odf_deg=cli_args.odf_deg, odf_scale_um=s, ram=ram)
                 for s in cli_args.odf_res)

    # print ODF analysis time
    print_analysis_time(start_time)


def save_frangi_arrays(fiber_dset_path, fiber_vec_img, fiber_vec_clr, frac_anis_img, frangi_img,
                       fiber_msk, soma_msk, px_size, save_dir, img_name):
    """
    Save the output arrays of the Frangi filter stage to TIF files.

    Parameters
    ----------
    fiber_dset_path: str
        path to initialized fiber orientation HDF5 dataset
        (not fitting the available RAM)

    fiber_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    fiber_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    frac_anis_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced volume image (fiber probability)

    fiber_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    soma_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
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

    # move large fiber orientation dataset to saving directory
    if fiber_dset_path is not None:
        shutil.move(fiber_dset_path, path.join(save_dir, 'fiber_vec_{0}.h5'.format(img_name)))
    # or save orientation vectors to NumPy file
    else:
        save_array('fiber_vec_{0}'.format(img_name), save_dir, fiber_vec_img, fmt='npy')

    # save orientation color map to TIFF
    save_array('fiber_cmap_{0}'.format(img_name), save_dir, fiber_vec_clr, px_size)

    # save fractional anisotropy map to TIFF
    save_array('frac_anis_{0}'.format(img_name), save_dir, frac_anis_img, px_size)

    # save Frangi-enhanced fiber volume to TIFF
    save_array('frangi_{0}'.format(img_name), save_dir, frangi_img, px_size)

    # save masked fiber volume to TIFF
    save_array('fiber_msk_{0}'.format(img_name), save_dir, fiber_msk, px_size)

    # save masked soma volume to TIFF
    if soma_msk is not None:
        save_array('soma_msk_{0}'.format(img_name), save_dir, soma_msk, px_size)


def save_odf_arrays(odf, bg, odi_pri, odi_sec, odi_tot, odi_anis, px_size, save_dir, img_name, odf_scale_um):
    """
    Save the output arrays of the ODF analysis stage to TIF and Nifti files.
    Arrays tagged with 'mrtrixview' are preliminarily transformed
    so that ODF maps viewed in MRtrix3 are spatially consistent
    with the analyzed microscopy volume, and the output TIF files.

    Parameters
    ----------
    odf: NumPy memory-map object or HDF5 dataset (axis order=(X,Y,Z,C), dtype=float32)
        ODF spherical harmonics coefficients

    bg: NumPy memory-map object or HDF5 dataset (axis order=(X,Y,Z), dtype=uint8)
        background for ODF visualization in MRtrix3

    odi_pri: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        primary orientation dispersion parameter

    odi_sec: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        secondary orientation dispersion parameter

    odi_tot: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        total orientation dispersion parameter

    odi_anis: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
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
    save_array('bg_mrtrixview_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, bg, fmt='nii')
    save_array('odf_mrtrixview_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, odf, fmt='nii')
    save_array('odi_pri_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, odi_pri, px_size, odi=True)
    save_array('odi_sec_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, odi_sec, px_size, odi=True)
    save_array('odi_tot_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, odi_tot, px_size, odi=True)
    save_array('odi_anis_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, odi_anis, px_size, odi=True)
