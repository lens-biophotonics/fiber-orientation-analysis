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
from foa3d.slicing import (config_frangi_batch, generate_slice_lists,
                           crop, crop_lst, slice_image)
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
    fa: numpy.ndarray (shape=(3,), dtype=float)
        fractional anisotropy
    """

    fa = np.sqrt(0.5 * divide_nonzero(
                 np.square((eigenval[..., 0] - eigenval[..., 1])) +
                 np.square((eigenval[..., 0] - eigenval[..., 2])) +
                 np.square((eigenval[..., 1] - eigenval[..., 2])),
                 np.sum(eigenval ** 2, axis=-1)))

    return fa


def init_frangi_volumes(img_shp, slc_shp, rsz_ratio, tmp_dir, z_rng=(0, None), msk_bc=False, ram=None):
    """
    Initialize the output datasets of the Frangi filtering stage.

    Parameters
    ----------
    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    tmp_dir: str
        temporary file directory

    z_rng: int
        output z-range in [px]

    msk_bc: bool
        if True, mask neuronal bodies
        in the optionally provided image channel

    ram: float
        maximum RAM available to the Frangi filtering stage [B]

    Returns
    -------
    fbr_dset_path: str
        path to initialized fiber orientation HDF5 dataset
        (not fitting the available RAM)

    fbr_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        initialized fiber orientation volume image

    fbr_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        initialized orientation colormap image

    fa_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized Frangi-enhanced image

    fbr_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized fiber mask image

    iso_fbr_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized fiber image (isotropic resolution)

    bc_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        initialized soma mask image

    z_sel: NumPy slice object
        selected z-depth range
    """

    # shape copies
    img_shp = img_shp.copy()
    slc_shp = slc_shp.copy()

    # adapt output z-axis shape if required
    z_min, z_max = z_rng
    if z_min != 0 or z_max is not None:
        if z_max is None:
            z_max = slc_shp[0]
        img_shp[0] = z_max - z_min
    z_sel = slice(z_min, z_max, 1)

    # output shape
    img_dims = len(img_shp)
    tot_shape = tuple(np.ceil(rsz_ratio * img_shp).astype(int))
    slc_shp[0] = tot_shape[0]

    # fiber channel arrays
    iso_fbr_img, _ = init_volume(tot_shape, dtype='uint8', chunks=slc_shp, name='iso_fiber', tmp=tmp_dir, ram=ram)
    frangi_img, _ = init_volume(tot_shape, dtype='uint8', chunks=slc_shp, name='frangi', tmp=tmp_dir, ram=ram)
    fbr_msk, _ = init_volume(tot_shape, dtype='uint8', chunks=slc_shp, name='fbr_msk', tmp=tmp_dir, ram=ram)
    fa_img, _ = init_volume(tot_shape, dtype='uint8', chunks=slc_shp, name='fa', tmp=tmp_dir, ram=ram)

    # soma channel array
    if msk_bc:
        bc_msk, _ = init_volume(tot_shape, dtype='uint8', chunks=slc_shp, name='bc_msk', tmp=tmp_dir, ram=ram)
    else:
        bc_msk = None

    # fiber orientation arrays
    vec_shape = tot_shape + (img_dims,)
    vec_slice_shape = tuple(slc_shp) + (img_dims,)
    fbr_vec_img, fbr_dset_path = \
        init_volume(vec_shape, dtype='float32', chunks=vec_slice_shape, name='fiber_vec', tmp=tmp_dir, ram=ram)
    fbr_vec_clr, _ = \
        init_volume(vec_shape, dtype='uint8', chunks=vec_slice_shape, name='fiber_cmap', tmp=tmp_dir, ram=ram)

    return fbr_dset_path, fbr_vec_img, fbr_vec_clr, fa_img, \
        frangi_img, fbr_msk, iso_fbr_img, bc_msk, z_sel


def init_odf_volumes(vec_img_shp, tmp_dir, odf_scale, odf_degrees=6, ram=None):
    """
    Initialize the output datasets of the ODF analysis stage.

    Parameters
    ----------
    vec_img_shp: tuple
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

    disarray: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=float32)
        local angular disarray

    vec_tensor_eigen: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        initialized array of fiber orientation tensor eigenvalues
    """

    # ODI maps shape
    odi_shp = tuple(np.ceil(np.divide(vec_img_shp, odf_scale)).astype(int))

    # create downsampled background memory map
    bg_shp = tuple(np.flip(odi_shp))
    bg_mrtrix, _ = init_volume(bg_shp, dtype='uint8', chunks=tuple(bg_shp[:2]) + (1,),
                               name='bg_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)

    # create ODF memory map
    nc = get_sph_harm_ncoeff(odf_degrees)
    odf_shp = odi_shp + (nc,)
    odf, _ = init_volume(odf_shp, dtype='float32', chunks=(1, 1, 1, nc),
                         name='odf_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)

    # create orientation tensor memory map
    vec_tensor_shape = odi_shp + (3,)
    vec_tensor_eigen, _ = init_volume(vec_tensor_shape, dtype='float32',
                                      chunks=(1, 1, 1, 3), name='tensor_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)

    # create ODI memory maps
    odi_pri, _ = init_volume(odi_shp, dtype='float32',
                             name='odi_pri_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)
    odi_sec, _ = init_volume(odi_shp, dtype='float32',
                             name='odi_sec_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)
    odi_tot, _ = init_volume(odi_shp, dtype='float32',
                             name='odi_tot_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)
    odi_anis, _ = init_volume(odi_shp, dtype='float32',
                              name='odi_anis_tmp{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)

    # create disarray map
    disarray, _ = init_volume(odi_shp, dtype='float32',
                              name='disarray{0}'.format(odf_scale), tmp=tmp_dir, ram=ram)

    return odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, disarray, vec_tensor_eigen


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
    vol_sz = item_sz * np.prod(shape).astype(np.float64)

    # create memory-mapped array or HDF5 file depending on available memory resources
    if vol_sz >= ram:
        vol, hdf5_path = create_hdf5_dset(shape, dtype, chunks=chunks, name=name, tmp=tmp)
    else:
        vol = create_memory_map(shape, dtype, name=name, tmp_dir=tmp, mmap_mode=mmap_mode)
        hdf5_path = None

    return vol, hdf5_path


def fiber_analysis(img, in_rng, bc_rng, out_rng, pad, ovlp, smooth_sigma, scales_px, px_rsz_ratio, z_sel,
                   fbr_vec_img, fbr_vec_clr, fa_img, frangi_img, iso_fbr_img, fbr_msk, bc_msk, ts_msk=None,
                   bc_ch=0, fb_ch=1, alpha=0.05, beta=1, gamma=100, dark=False, msk_bc=False,
                   hsv_vec_cmap=False, pad_mode='reflect', is_tiled=False, fb_thr='li', bc_thr='yen', ts_thr=0.005):
    """
    Conduct a Frangi-based fiber orientation analysis on basic slices selected from the whole microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X))
        fiber fluorescence volume image

    in_rng: NumPy slice object
        input image range (fibers)

    bc_rng: NumPy slice object
        input image range (neurons)

    out_rng: NumPy slice object
        output range

    pad: numpy.ndarray (axis order=(Z,Y,X))
        image padding array

    ovlp: int
        overlapping range between slices along each axis [px]

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [px]

    scales_px: numpy.ndarray (dtype=int)
        spatial scales [px]

    px_rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    z_sel: NumPy slice object
        selected z-depth range

    fbr_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    fbr_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    fa_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced volume image (fiber probability volume)

    iso_fbr_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fbr_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    bc_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        soma mask image

    tissue_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    bc_ch: int
        neuronal bodies channel

    fb_ch: int
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
        if True, generate an HSV orientation color map based on XY-plane orientation angles
        (instead of an RGB map using the cartesian components of the estimated vectors)

    msk_bc: bool
        if True, mask neuronal bodies exploiting the autofluorescence
        signal of lipofuscin pigments

    pad_mode: str
        image padding mode

    is_tiled: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    fb_thr: str
         thresholding method applied to the myelinated fibers channel

    bc_thr: str
        thresholding method applied to the neuronal bodies channel

    ts_thr: float
        relative threshold on non-zero tissue pixels

    Returns
    -------
    None
    """

    # slice fiber image slice
    fbr_slc, ts_msk_slc = slice_image(img, in_rng, fb_ch, ts_msk=ts_msk, is_tiled=is_tiled)

    # skip background slice
    crt = np.count_nonzero(ts_msk_slc) / np.prod(ts_msk_slc.shape) > ts_thr \
        if ts_msk_slc is not None else np.max(fbr_slc) != 0
    if crt:

        # preprocess fiber slice
        iso_fbr_slc, pad_rsz, ts_msk_slc_rsz = \
            correct_image_anisotropy(fbr_slc, px_rsz_ratio, sigma=smooth_sigma, pad=pad, ts_msk=ts_msk_slc)

        # pad fiber slice if required
        if pad_rsz is not None:
            iso_fbr_slc = np.pad(iso_fbr_slc, pad_rsz, mode=pad_mode)

            # pad tissue mask if available
            if ts_msk_slc_rsz is not None:
                ts_msk_slc_rsz = np.pad(ts_msk_slc_rsz, pad_rsz, mode='constant')

        # 3D Frangi filter
        frangi_slc, fbr_vec_slc, eigenval_slc = \
            frangi_filter(iso_fbr_slc, scales_px=scales_px, alpha=alpha, beta=beta, gamma=gamma, dark=dark)

        # crop resulting slices
        iso_fbr_slc, frangi_slc, fbr_vec_slc, eigenval_slc, ts_msk_slc_rsz = \
            crop_lst([iso_fbr_slc, frangi_slc, fbr_vec_slc, eigenval_slc, ts_msk_slc_rsz], out_rng, ovlp)

        # generate fractional anisotropy image
        fa_slc = compute_fractional_anisotropy(eigenval_slc)

        # generate fiber orientation color map
        fbr_clr_slc = hsv_orient_cmap(fbr_vec_slc) if hsv_vec_cmap else rgb_orient_cmap(fbr_vec_slc)

        # remove background
        fbr_vec_slc, fbr_clr_slc, fa_slc, fiber_msk_slice = \
            mask_background(frangi_slc, fbr_vec_slc, fbr_clr_slc, fa_slc,
                            ts_msk=ts_msk_slc_rsz, method=fb_thr, invert=False)

        # (optional) neuronal body masking
        if msk_bc:

            # get soma image slice
            bc_slc = slice_image(img, bc_rng, bc_ch, is_tiled=is_tiled)

            # resize soma slice (lateral blurring and downsampling)
            iso_bc_slc, _, _ = correct_image_anisotropy(bc_slc, px_rsz_ratio)

            # crop isotropized soma slice
            iso_bc_slc = crop(iso_bc_slc, out_rng)

            # mask neuronal bodies
            fbr_vec_slc, fbr_clr_slc, fa_slc, bc_msk_slc = \
                mask_background(iso_bc_slc, fbr_vec_slc, fbr_clr_slc, fa_slc, method=bc_thr, invert=True)

        else:
            bc_msk_slc = None

        # fill memory-mapped output arrays
        fill_frangi_volumes(fbr_vec_img, fbr_vec_clr, fa_img, frangi_img, iso_fbr_img, fbr_msk, bc_msk,
                            fbr_vec_slc, fbr_clr_slc, fa_slc, frangi_slc, iso_fbr_slc,
                            fiber_msk_slice, bc_msk_slc, out_rng, z_sel)


def fill_frangi_volumes(fbr_vec_img, fbr_vec_clr, fa_img, frangi_img, iso_fbr_img, fbr_msk, bc_msk, fbr_vec_slc,
                        fbr_clr_slc, fa_slc, frangi_slc, iso_fbr_slc, fbr_msk_slc, bc_msk_slc, out_rng, z_sel):
    """
    Fill the memory-mapped output arrays of the Frangi filter stage.

    Parameters
    ----------
    fbr_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    fbr_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    fa_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced image (fiber probability image)

    iso_fbr_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    fbr_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    bc_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        soma mask image

    fbr_vec_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        fiber orientation vector image slice

    fbr_clr_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        orientation colormap image slice

    fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        fractional anisotropy image slice

    frangi_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float32)
        Frangi-enhanced image slice

    iso_fbr_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image slice

    fbr_msk_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image slice

    bc_msk_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        soma mask image slice

    out_rng: tuple
        3D slice output index range

    z_sel: NumPy slice object
        selected z-depth range

    Returns
    -------
    None
    """

    # fill memory-mapped output arrays
    vec_rng_out = tuple(np.append(out_rng, slice(0, 3, 1)))
    fbr_vec_img[vec_rng_out] = fbr_vec_slc[z_sel, ...]
    fbr_vec_clr[vec_rng_out] = fbr_clr_slc[z_sel, ...]
    iso_fbr_img[out_rng] = iso_fbr_slc[z_sel, ...].astype(np.uint8)
    frangi_img[out_rng] = (255 * frangi_slc[z_sel, ...]).astype(np.uint8)
    fa_img[out_rng] = (255 * fa_slc[z_sel, ...]).astype(np.uint8)
    fbr_msk[out_rng] = (255 * (1 - fbr_msk_slc[z_sel, ...])).astype(np.uint8)

    # fill memory-mapped output soma mask, if available
    if bc_msk is not None:
        bc_msk[out_rng] = (255 * bc_msk_slc[z_sel, ...]).astype(np.uint8)


def mask_background(img, fbr_vec_slc, fbr_clr_slc, fa_slc=None, method='yen', invert=False, ts_msk=None):
    """
    Mask fiber orientation arrays.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        fiber (or neuron) fluorescence volume image

    fbr_vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        fiber orientation vector slice

    fbr_clr_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        fiber orientation colormap slice

    fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        fractional anisotropy slice

    method: str
        thresholding method (refer to skimage.filters)

    invert: bool
        mask inversion flag

    ts_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    Returns
    -------
    fbr_vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        orientation vector patch (masked)

    orientcol_slice: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap patch (masked)

    fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        fractional anisotropy patch (masked)

    bg: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
        background mask
    """

    # generate background mask
    bg = create_background_mask(img, method=method)

    # apply tissue reconstruction mask, when provided
    if ts_msk is not None:
        bg = np.logical_or(bg, np.logical_not(ts_msk))

    # invert mask
    if invert:
        bg = np.logical_not(bg)

    # apply mask to input arrays
    fbr_vec_slc[bg, :] = 0
    fbr_clr_slc[bg, :] = 0
    fa_slc[bg] = 0

    return fbr_vec_slc, fbr_clr_slc, fa_slc, bg


def odf_analysis(fbr_vec_img, iso_fbr_img, px_sz_iso, save_dir, tmp_dir, img_name, odf_scale_um, odf_norm,
                 odf_deg=6, ram=None):
    """
    Estimate 3D fiber ODFs from basic orientation data chunks using parallel threads.

    Parameters
    ----------
    fbr_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vectors

    iso_fbr_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber volume

    px_sz_iso: numpy.ndarray (shape=(3,), dtype=float)
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
    odf_scale = int(np.ceil(odf_scale_um / px_sz_iso[0]))

    # initialize ODF analysis output volumes
    odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, disarray, tensor \
        = init_odf_volumes(fbr_vec_img.shape[:-1], tmp_dir, odf_scale=odf_scale, odf_degrees=odf_deg, ram=ram)

    # generate downsampled background for MRtrix3 mrview
    bg_img = fbr_vec_img if iso_fbr_img is None else iso_fbr_img
    generate_odf_background(bg_img, bg_mrtrix, vxl_side=odf_scale)

    # compute ODF coefficients
    odf = compute_odf_map(fbr_vec_img, odf, odi_pri, odi_sec, odi_tot, odi_anis, disarray, tensor,
                          odf_scale, odf_norm, odf_deg=odf_deg)

    # save memory maps to file
    save_odf_arrays(odf, bg_mrtrix, odi_pri, odi_sec, odi_tot, odi_anis, disarray,
                    px_sz_iso, save_dir, img_name, odf_scale_um)


def parallel_frangi_on_slices(img, cli_args, save_dir, tmp_dir, img_name, ts_msk=None,
                              ram=None, jobs=4, backend='threading', is_tiled=False, verbose=10):
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

    tissue_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

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
    fbr_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    iso_fbr_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    px_sz_iso: int
        isotropic pixel size [μm]

    img_name: str
        name of the input microscopy image
    """

    # get Frangi filter configuration
    alpha, beta, gamma, frangi_sigma, frangi_sigma_um, smooth_sigma, px_sz, px_sz_iso, \
        z_rng, bc_ch, fb_ch, msk_bc, hsv_vec_cmap, img_name = get_frangi_config(cli_args, img_name)

    # get info about the input microscopy image
    img_shp, img_shp_um, img_item_sz, fb_ch, msk_bc = get_image_info(img, px_sz, msk_bc, fb_ch, is_tiled=is_tiled)

    # configure the batch of basic image slices analyzed using parallel threads
    batch_sz, in_slc_shp, in_slc_shp_um, px_rsz_ratio, ovlp, ovlp_rsz = \
        config_frangi_batch(px_sz, px_sz_iso, img_shp, img_item_sz, smooth_sigma, frangi_sigma_um, ram=ram, jobs=jobs)

    # get info about the processed image slices
    in_rng_lst, in_pad_lst, out_rng_lst, bc_rng_lst, out_slc_shp, tot_slc_num, batch_sz = \
        generate_slice_lists(in_slc_shp, img_shp, batch_sz, px_rsz_ratio, ovlp, msk_bc=msk_bc, jobs=jobs)

    # initialize output arrays
    fbr_dset_path, fbr_vec_img, fbr_vec_clr, fa_img, frangi_img, fbr_msk, iso_fbr_img, bc_msk, z_sel = \
        init_frangi_volumes(img_shp, out_slc_shp, px_rsz_ratio, tmp_dir, z_rng=z_rng, msk_bc=msk_bc, ram=ram)

    # print Frangi filter configuration
    print_frangi_info(alpha, beta, gamma, frangi_sigma_um, img_shp_um, in_slc_shp_um, tot_slc_num,
                      px_sz, img_item_sz, msk_bc)

    # parallel Frangi filter-based fiber orientation analysis of microscopy image slices
    start_time = perf_counter()
    with Parallel(n_jobs=batch_sz, backend=backend, verbose=verbose, max_nbytes=None) as parallel:
        parallel(
            delayed(fiber_analysis)(img, in_rng_lst[i], bc_rng_lst[i], out_rng_lst[i], in_pad_lst[i], ovlp_rsz,
                                    smooth_sigma, frangi_sigma, px_rsz_ratio, z_sel, fbr_vec_img, fbr_vec_clr,
                                    fa_img, frangi_img, iso_fbr_img, fbr_msk, bc_msk, ts_msk=ts_msk,
                                    bc_ch=bc_ch, fb_ch=fb_ch, alpha=alpha, beta=beta, gamma=gamma,
                                    msk_bc=msk_bc, hsv_vec_cmap=hsv_vec_cmap, is_tiled=is_tiled)
            for i in range(tot_slc_num))

    # save output arrays
    save_frangi_arrays(fbr_dset_path, fbr_vec_img, fbr_vec_clr, fa_img, frangi_img, fbr_msk, bc_msk, px_sz_iso,
                       save_dir, img_name)

    # print Frangi filtering time
    print_analysis_time(start_time)

    return fbr_vec_img, iso_fbr_img, px_sz_iso, img_name


def parallel_odf_at_scales(fbr_vec_img, iso_fbr_img, cli_args, px_sz_iso, save_dir, tmp_dir, img_name,
                           backend='loky', ram=None, verbose=100):
    """
    Iterate over the required spatial scales and apply the parallel ODF analysis
    implemented in parallel_odf_on_slices().

    Parameters
    ----------
    fbr_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    iso_fbr_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    px_sz_iso: numpy.ndarray (axis order=(3,), dtype=float)
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
    if px_sz_iso is None:
        px_sz_iso = cli_args.px_size_z * np.ones((3,))

    # parallel ODF analysis of fiber orientation vectors
    # over the required spatial scales
    n_jobs = min(num_cpu, len(cli_args.odf_res))
    with Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose, max_nbytes=None) as parallel:
        parallel(delayed(odf_analysis)(fbr_vec_img, iso_fbr_img, px_sz_iso, save_dir, tmp_dir, img_name,
                                       odf_norm=odf_norm, odf_deg=cli_args.odf_deg, odf_scale_um=s, ram=ram)
                 for s in cli_args.odf_res)

    # print ODF analysis time
    print_analysis_time(start_time)


def save_frangi_arrays(fbr_dset_path, fbr_vec_img, fbr_vec_clr, fa_img, frangi_img,
                       fbr_msk, bc_msk, px_sz, save_dir, img_name):
    """
    Save the output arrays of the Frangi filter stage to TIF files.

    Parameters
    ----------
    fbr_dset_path: str
        path to initialized fiber orientation HDF5 dataset
        (not fitting the available RAM)

    fbr_vec_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector image

    fbr_vec_clr: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X,C), dtype=uint8)
        orientation colormap image

    fa_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fractional anisotropy image

    frangi_img: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced volume image (fiber probability)

    fbr_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image

    bc_msk: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=uint8)
        neuron mask image

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size (Z,Y,X) [μm]

    save_dir: str
        saving directory string path

    img_name: str
        name of the input microscopy image

    Returns
    -------
    None
    """

    # move large fiber orientation dataset to saving directory
    if fbr_dset_path is not None:
        shutil.move(fbr_dset_path, path.join(save_dir, 'fiber_vec_{0}.h5'.format(img_name)))
    # or save orientation vectors to TIFF
    else:
        save_array('fiber_vec_{0}'.format(img_name), save_dir, np.moveaxis(fbr_vec_img, -1, 1), px_sz)

    # save orientation color map to TIFF
    save_array('fiber_cmap_{0}'.format(img_name), save_dir, fbr_vec_clr, px_sz)

    # save fractional anisotropy map to TIFF
    save_array('frac_anis_{0}'.format(img_name), save_dir, fa_img, px_sz)

    # save Frangi-enhanced fiber volume to TIFF
    save_array('frangi_{0}'.format(img_name), save_dir, frangi_img, px_sz)

    # save masked fiber volume to TIFF
    save_array('fiber_msk_{0}'.format(img_name), save_dir, fbr_msk, px_sz)

    # save masked soma volume to TIFF
    if bc_msk is not None:
        save_array('soma_msk_{0}'.format(img_name), save_dir, bc_msk, px_sz)


def save_odf_arrays(odf, bg, odi_pri, odi_sec, odi_tot, odi_anis, disarray, px_sz, save_dir, img_name, odf_scale_um):
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

    odi_pri: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=float32)
        primary orientation dispersion parameter

    odi_sec: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=float32)
        secondary orientation dispersion parameter

    odi_tot: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=float32)
        total orientation dispersion parameter

    odi_anis: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=float32)
        orientation dispersion anisotropy parameter

    disarray: NumPy memory-map object or HDF5 dataset (axis order=(Z,Y,X), dtype=float32)

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
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
    save_array('odi_pri_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, odi_pri, px_sz, odi=True)
    save_array('odi_sec_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, odi_sec, px_sz, odi=True)
    save_array('odi_tot_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, odi_tot, px_sz, odi=True)
    save_array('odi_anis_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, odi_anis, px_sz, odi=True)
    save_array('disarray_sv{0}_{1}'.format(odf_scale_um, img_name), save_dir, disarray, px_sz, odi=True)
