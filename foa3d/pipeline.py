from time import perf_counter

import numpy as np
from joblib import Parallel, delayed

from foa3d.frangi import (frangi_filter, init_frangi_arrays, mask_background,
                          write_frangi_arrays)
from foa3d.input import get_frangi_config, get_resource_config
from foa3d.odf import (compute_odf_map, generate_odf_background,
                       init_odf_arrays)
from foa3d.output import save_frangi_arrays, save_odf_arrays
from foa3d.preprocessing import correct_anisotropy
from foa3d.printing import (print_flsh, print_frangi_info, print_odf_info,
                            print_frangi_progress)
from foa3d.slicing import (check_background, config_frangi_batch,
                           generate_slice_lists, crop, crop_lst, slice_image)
from foa3d.spharm import get_sph_harm_norm_factors
from foa3d.utils import get_available_cores


def parallel_frangi_over_slices(cli_args, save_dirs, in_img):
    """
    Apply 3D Frangi filtering to basic TPFM image slices using parallel threads.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    save_dirs: str
        saving directories

    in_img: dict
        input image dictionary

            data: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
                3D microscopy image

            ts_msk: numpy.ndarray (dtype=bool)
                tissue reconstruction binary mask

            ch_ax: int
                RGB image channel axis (either 1, 3, or None for grayscale images)

            fb_ch: int
                neuronal fibers channel

            bc_ch: int
                brain cell soma channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
                3D FWHM of the PSF [μm]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            name: str
                name of the 3D microscopy image

            is_vec: bool
                vector field flag

            shape: numpy.ndarray (shape=(3,), dtype=int)
                total image shape

            shape_um: numpy.ndarray (shape=(3,), dtype=float)
                total image shape [μm]

            item_sz: int
                image item size [B]

    Returns
    -------
    out_img: dict
        output image dictionary

        fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
            fiber orientation vector field

        fbr_clr: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=uint8)
            orientation colormap image

        fa: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            fractional anisotropy image

        frangi: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            Frangi-enhanced image (fiber probability image)

        iso_fbr_img: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            isotropic fiber image

        fbr_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            fiber mask image

        bc_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
            soma mask image

        px_sz: numpy.ndarray (shape=(3,), dtype=float)
            output pixel size [μm]
    """
    # skip Frangi filter stage if orientation vectors were directly provided as input
    if in_img['is_vec']:
        out_img = dict()
        out_img['fbr_vec'] = in_img['data']
        out_img['iso_fbr'] = None
        out_img['px_sz'] = None

    else:

        # get resources configuration
        ram, jobs = get_resource_config(cli_args)

        # get Frangi filter configuration
        frangi_cfg, px_sz_iso = get_frangi_config(cli_args, in_img)

        # configure the batch of basic image slices analyzed using parallel threads
        batch_sz, slc_shp, slc_shp_um, rsz_ratio, ovlp, ovlp_rsz = \
            config_frangi_batch(in_img, frangi_cfg, px_sz_iso, ram=ram, jobs=jobs)

        # generate image range lists
        slc_rng, out_slc_shp, tot_slc, batch_sz = \
            generate_slice_lists(slc_shp, in_img['shape'], batch_sz, rsz_ratio, ovlp, in_img['msk_bc'])

        # initialize output arrays
        out_img, z_sel = init_frangi_arrays(frangi_cfg, in_img['shape'], out_slc_shp, rsz_ratio, save_dirs['tmp'],
                                            msk_bc=in_img['msk_bc'])

        # print Frangi filter configuration
        print_frangi_info(in_img, frangi_cfg, slc_shp_um, tot_slc)

        # parallel Frangi filter-based fiber orientation analysis of microscopy image slices
        print_flsh(f"[Parallel(n_jobs={batch_sz})]: Using backend ThreadingBackend with {batch_sz} concurrent workers.")
        t_start = perf_counter()
        with Parallel(n_jobs=batch_sz, prefer='threads', require='sharedmem') as parallel:
            parallel(
                delayed(frangi_analysis)(s, in_img, out_img, frangi_cfg, ovlp_rsz, rsz_ratio, z_sel, **slc_rng,
                                         print_info=(t_start, batch_sz, tot_slc))
                for s in range(tot_slc))

        # save output arrays
        save_frangi_arrays(save_dirs['frangi'], in_img['name'], out_img, px_sz_iso, ram=ram)

        # add isotropic pixel size
        out_img['px_sz'] = px_sz_iso

    return out_img


def frangi_analysis(s, in_img, out_img, cfg, ovlp, rsz_ratio, z_sel, in_rng, bc_rng, out_rng, in_pad,
                    print_info=None, pad_mode='reflect', bc_thr='yen', ts_thr=0.0001, verbose=10):
    """
    Conduct a Frangi-based fiber orientation analysis on basic slices selected from the whole microscopy volume image.

    Parameters
    ----------
    s: int
        image slice index

    in_img: dict
        input image dictionary (extended)

            data: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
                3D microscopy image

            ts_msk: numpy.ndarray (dtype=bool)
                tissue reconstruction binary mask

            ch_ax: int
                RGB image channel axis (either 1, 3, or None for grayscale images)

            fb_ch: int
                neuronal fibers channel

            bc_ch: int
                brain cell soma channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
                3D FWHM of the PSF [μm]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            name: str
                name of the 3D microscopy image

            is_vec: bool
                vector field flag

            shape: numpy.ndarray (shape=(3,), dtype=int)
                total image shape

            shape_um: numpy.ndarray (shape=(3,), dtype=float)
                total image shape [μm]

            item_sz: int
                image item size [B]

    out_img: dict
        output image dictionary

            vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
                fiber orientation vector field

            clr: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap image

            fa: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                fractional anisotropy image

            frangi: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                Frangi-enhanced image (fiber probability image)

            iso: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                isotropic fiber image

            fbr_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                fiber mask image

            bc_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                soma mask image

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                output pixel size [μm]

    cfg: dict
        Frangi filter configuration

            alpha: float
                plate-like score sensitivity

            beta: float
                blob-like score sensitivity

            gamma: float
                background score sensitivity

            scales_px: numpy.ndarray (dtype=float)
                Frangi filter scales [px]

            scales_um: numpy.ndarray (dtype=float)
                Frangi filter scales [μm]

            smooth_sd: numpy.ndarray (shape=(3,), dtype=int)
                3D standard deviation of the smoothing Gaussian filter [px]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            z_rng: int
                output z-range in [px]

            bc_ch: int
                neuronal bodies channel

            fb_ch: int
                myelinated fibers channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            hsv_cmap: bool
                generate HSV colormap of 3D fiber orientations

            exp_all: bool
                export all images

    ovlp: int
        overlapping range between slices along each axis [px]

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    z_sel: NumPy slice object
        selected z-depth range

    in_rng: NumPy slice object
        input image range (fibers)

    bc_rng: NumPy slice object
        input image range (neurons)

    out_rng: NumPy slice object
        output range

    in_pad: numpy.ndarray (axis order=(Z,Y,X))
        image padding range

    print_info: tuple
        optional printed information

    pad_mode: str
        image padding mode

    bc_thr: str
        thresholding method applied to the neuronal bodies channel

    ts_thr: float
        relative threshold on non-zero tissue pixels

    verbose: int
        verbosity level (print progress every N=verbose image slices)

    Returns
    -------
    None
    """
    # select fiber image slice
    fbr_slc, ts_msk_slc = slice_image(in_img['data'], in_rng[s], in_img['ch_ax'], ch=in_img['fb_ch'],
                                      ts_msk=in_img['ts_msk'])

    # process non-background image slice
    is_valid = check_background(fbr_slc, ts_msk=ts_msk_slc, ts_thr=ts_thr)
    if is_valid:
        orient_slc, frangi_slc, iso_slc, fbr_msk_slc = \
            analyze_fibers(fbr_slc, cfg, ovlp, rsz_ratio, out_rng[s], in_pad[s], pad_mode=pad_mode, ts_msk=ts_msk_slc)

        # (optional) neuronal body masking
        if in_img['msk_bc']:

            # get soma image slice
            bc_slc, _ = slice_image(in_img['data'], bc_rng[s], in_img['ch_ax'], ch=in_img['bc_ch'])

            # suppress soma contribution
            orient_slc, bc_msk_slc = reject_brain_cells(bc_slc, orient_slc, rsz_ratio, out_rng[s], bc_thr)

        # soma mask not available
        else:
            bc_msk_slc = None

        # fill memory-mapped output arrays
        write_frangi_arrays(out_rng[s], z_sel, iso_slc, frangi_slc, fbr_msk_slc, bc_msk_slc, **orient_slc, **out_img)

    # print progress
    if print_info is not None:
        print_frangi_progress(print_info, is_valid, verbose=verbose)


def analyze_fibers(fbr_slc, cfg, ovlp, rsz_ratio, out_rng, pad, pad_mode='reflect', ts_msk=None):
    """
    Analyze 3D fiber orientations exploiting a Frangi-filter-based
    unsupervised enhancement of tubular structures.

    Parameters
    ----------
    fbr_slc: numpy.ndarray (axis order=(Z,Y,X))
        fiber image slice

    cfg: dict
        Frangi filter configuration

            alpha: float
                plate-like score sensitivity

            beta: float
                blob-like score sensitivity

            gamma: float
                background score sensitivity

            scales_px: numpy.ndarray (dtype=float)
                Frangi filter scales [px]

            scales_um: numpy.ndarray (dtype=float)
                Frangi filter scales [μm]

            smooth_sd: numpy.ndarray (shape=(3,), dtype=int)
                3D standard deviation of the smoothing Gaussian filter [px]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            z_rng: int
                output z-range in [px]

            bc_ch: int
                neuronal bodies channel

            fb_ch: int
                myelinated fibers channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            hsv_cmap: bool
                generate HSV colormap of 3D fiber orientations

            exp_all: bool
                export all images

    ovlp: int
        overlapping range between slices along each axis [px]

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    out_rng: NumPy slice object
        output range

    pad: numpy.ndarray (axis order=(Z,Y,X))
        image padding range

    pad_mode: str
        image padding mode

    ts_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    Returns
    -------
    orient_slc: dict
        slice orientation dictionary

        vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
            3D fiber orientation field

        clr_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
            orientation colormap image

        fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
            fractional anisotropy image

    frangi: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        Frangi-enhanced image slice (fiber probability image)

    iso_fbr_img: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image slice

    fbr_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        fiber mask image slice
    """
    # preprocess fiber slice (lateral blurring and downsampling)
    iso_fbr_slc, pad_rsz, ts_msk_slc_rsz = \
        correct_anisotropy(fbr_slc, rsz_ratio, sigma=cfg['smooth_sd'], pad=pad, ts_msk=ts_msk)

    # pad fiber slice if required
    if pad_rsz is not None:
        iso_fbr_slc = np.pad(iso_fbr_slc, pad_rsz, mode=pad_mode)

        # pad tissue mask if available
        if ts_msk_slc_rsz is not None:
            ts_msk_slc_rsz = np.pad(ts_msk_slc_rsz, pad_rsz, mode='constant')

    # 3D Frangi filter
    orient_slc, frangi_slc = \
        frangi_filter(iso_fbr_slc, scales_px=cfg['scales_px'],
                      alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'], hsv=cfg['hsv_cmap'])

    # crop resulting slices
    orient_slc, frangi_slc, iso_fbr_slc, ts_msk_slc_rsz = \
        crop_lst([orient_slc, frangi_slc, iso_fbr_slc, ts_msk_slc_rsz], out_rng, ovlp)

    # remove Frangi filter background
    orient_slc, fbr_msk_slc = \
        mask_background(frangi_slc, orient_slc, ts_msk=ts_msk_slc_rsz, method=cfg['fb_thr'], invert=False)

    return orient_slc, frangi_slc, iso_fbr_slc, fbr_msk_slc


def reject_brain_cells(bc_slc, orient_slc, rsz_ratio, out_rng, bc_thr='yen'):
    """
    Suppress soma contribution using the optional image channel
    of neuronal bodies.

    Parameters
    ----------
    bc_slc: numpy.ndarray (axis order=(Z,Y,X))
        brain cell soma image slice

    orient_slc: dict
        slice orientation dictionary

            vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
                3D fiber orientation field

            clr_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap image

            fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                fractional anisotropy image

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    out_rng: NumPy slice object
        output range

    bc_thr: str
        thresholding method applied to the neuronal bodies channel

    Returns
    -------
    orient_slc: dict
        (masked) slice orientation dictionary

        vec_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
            (masked) 3D fiber orientation field

        clr_slc: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
            (masked) orientation colormap image

        fa_slc: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
            (masked) fractional anisotropy image

    bc_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
        brain cell mask
    """
    # resize soma slice (lateral downsampling)
    iso_bc, _, _ = correct_anisotropy(bc_slc, rsz_ratio)

    # crop isotropic brain cell soma slice
    iso_bc = crop(iso_bc, out_rng)

    # mask neuronal bodies
    orient_slc, bc_msk = mask_background(iso_bc, orient_slc, method=bc_thr, invert=True)

    return orient_slc, bc_msk


def parallel_odf_over_scales(cli_args, save_dirs, fbr_vec, iso_fbr, px_sz, img_name, backend='loky', verbose=100):
    """
    Iterate over the required spatial scales and apply the parallel ODF analysis
    implemented in parallel_odf_on_slices().

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    save_dirs: dict
        saving directories
        ('frangi': Frangi filter, 'odf': ODF analysis, 'tmp': temporary files)

    fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector field

    iso_fbr: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    px_sz: numpy.ndarray (axis order=(3,), dtype=float)
        pixel size [μm]

    img_name: str
        name of the input volume image

    backend: str
        backend module employed by joblib.Parallel

    verbose: int
        joblib verbosity level

    Returns
    -------
    None
    """
    # print ODF analysis heading
    print_odf_info(cli_args.odf_res, cli_args.odf_deg)

    # export all images (optional)
    exp_all = cli_args.exp_all

    # compute spherical harmonics normalization factors (once for all scales)
    odf_norm = get_sph_harm_norm_factors(cli_args.odf_deg)

    # get number of logical cores
    num_cpu = get_available_cores()

    # get pixel size from CLI arguments if not provided
    if px_sz is None:
        px_sz = np.array([cli_args.px_size_z, cli_args.px_size_xy, cli_args.px_size_xy])

    # parallel ODF analysis of fiber orientation vectors
    # over spatial scales of interest
    n_scales = len(cli_args.odf_res)
    batch_sz = min(num_cpu, n_scales)
    with Parallel(n_jobs=batch_sz, backend=backend, verbose=verbose) as parallel:
        parallel(delayed(odf_analysis)(fbr_vec, iso_fbr, px_sz, save_dirs, img_name,
                                       odf_norm=odf_norm, odf_deg=cli_args.odf_deg, odf_scale_um=s, exp_all=exp_all)
                 for s in cli_args.odf_res)

    # print output directory
    print_flsh(f"\nODF and dispersion maps saved to: {save_dirs['odf']}\n")


def odf_analysis(fbr_vec, iso_fbr, px_sz, save_dirs, img_name, odf_scale_um, odf_norm, odf_deg=6, exp_all=False):
    """
    Estimate 3D fiber ODFs from basic orientation data chunks using parallel threads.

    Parameters
    ----------
    fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector field

    iso_fbr: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    save_dirs: dict
        saving directories
        ('frangi': Frangi filter, 'odf': ODF analysis, 'tmp': temporary files)

    img_name: str
        name of the 3D microscopy image

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    odf_norm: numpy.ndarray (dtype: float)
        2D array of spherical harmonic normalization factors

    odf_deg: int
        degrees of the spherical harmonic series expansion

    exp_all: bool
        export all images

    Returns
    -------
    None
    """
    # get the ODF kernel size in [px]
    odf_scale = int(np.ceil(odf_scale_um / px_sz[0]))

    # initialize ODF analysis output arrays
    odf, odi, bg_mrtrix, vec_tensor_eigen = \
        init_odf_arrays(fbr_vec.shape[:-1], save_dirs['tmp'], odf_scale=odf_scale, odf_deg=odf_deg, exp_all=exp_all)

    # generate downsampled background for MRtrix3 mrview
    generate_odf_background(bg_mrtrix, fbr_vec, vxl_side=odf_scale, iso_fbr=iso_fbr)

    # compute ODF coefficients
    odf = compute_odf_map(fbr_vec, odf, odi, vec_tensor_eigen, odf_scale, odf_norm, odf_deg=odf_deg)

    # save memory maps to TIFF or NIfTI files
    save_odf_arrays(save_dirs['odf'], img_name, odf_scale_um, px_sz, odf, bg_mrtrix, **odi)
