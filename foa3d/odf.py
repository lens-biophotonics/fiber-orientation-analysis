import numpy as np
import scipy as sp
from numba import njit
from skimage.transform import resize

from foa3d.spharm import fiber_vectors_to_sph_harm, get_sph_harm_ncoeff
from foa3d.utils import create_memory_map, normalize_image, transform_axes


def compute_odf_map(fbr_vec, px_sz, odf, odi, fbr_dnst, vec_tensor_eigen, vxl_side, odf_norm,
                    odf_deg=6, vxl_thr=0.5, vec_thr=0.000001):
    """
    Compute the spherical harmonics coefficients iterating over super-voxels
    of fiber orientation vectors.

    Parameters
    ----------
    fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float)
        fiber orientation vectors

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    odf: NumPy memory-map object (axis order=(X,Y,Z,C), dtype=float32)
                initialized array of ODF spherical harmonics coefficients

    odi: dict
        orientation dispersion dictionary

            odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                primary orientation dispersion index

            odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                secondary orientation dispersion index

            odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                total orientation dispersion index

            odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                orientation dispersion index anisotropy

    fbr_dnst

    vec_tensor_eigen: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        initialized array of orientation tensor eigenvalues

    vxl_side: int
        side of the ODF super-voxel [px]

    odf_norm: numpy.ndarray (dtype: float)
        2D array of spherical harmonics normalization factors

    odf_deg: int
        degrees of the spherical harmonics series expansion

    vxl_thr: float
        minimum relative threshold on the sliced voxel volume

    vec_thr: float
        minimum relative threshold on non-zero orientation vectors

    Returns
    -------
    odf: numpy.ndarray (axis order=(X,Y,Z,C), dtype=float32)
        volumetric map of real-valued spherical harmonics coefficients
    """
    # iterate over ODF super-voxels
    ref_vxl_size = min(vxl_side, fbr_vec.shape[0]) * vxl_side**2
    for z in range(0, fbr_vec.shape[0], vxl_side):
        zmax = z + vxl_side

        for y in range(0, fbr_vec.shape[1], vxl_side):
            ymax = y + vxl_side

            for x in range(0, fbr_vec.shape[2], vxl_side):
                xmax = x + vxl_side

                # slice orientation voxel
                vec_vxl = fbr_vec[z:zmax, y:ymax, x:xmax, :]
                zerovec = np.count_nonzero(np.all(vec_vxl == 0, axis=-1))
                sli_vxl_size = np.prod(vec_vxl.shape[:-1])

                # local fiber density estimate
                zv, yv, xv = z // vxl_side, y // vxl_side, x // vxl_side
                fbr_dnst[zv, yv, xv] = (sli_vxl_size - zerovec) / (sli_vxl_size * np.prod(px_sz))

                # skip boundary voxels and voxels without enough non-zero orientation vectors
                if sli_vxl_size / ref_vxl_size > vxl_thr and 1 - zerovec / sli_vxl_size > vec_thr:

                    # flatten orientation supervoxel
                    vec_arr = vec_vxl.ravel()

                    # compute ODF and orientation tensor eigenvalues
                    odf[zv, yv, xv, :] = fiber_vectors_to_sph_harm(vec_arr, odf_deg, odf_norm)
                    vec_tensor_eigen[zv, yv, xv, :] = compute_vec_tensor_eigen(vec_arr)

    # compute slice-wise dispersion and anisotropy parameters
    compute_orientation_dispersion(vec_tensor_eigen, **odi)

    # set dispersion background to 0
    mask_orientation_dispersion(vec_tensor_eigen, odi)

    # transform axes
    odf = transform_axes(odf, swapped=(0, 2), flipped=(1, 2))

    return odf, fbr_dnst


@njit(cache=True)
def compute_orientation_dispersion(vec_tensor_eigen, odi_pri, odi_sec, odi_tot, odi_anis):
    """
    Compute orientation dispersion parameters.

    Parameters
    ----------
    vec_tensor_eigen: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        orientation tensor eigenvalues
        computed from an ODF super-voxel

    odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        primary orientation dispersion index

    odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        secondary orientation dispersion index

    odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        total orientation dispersion index

    odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        orientation dispersion index anisotropy

    Returns
    -------
    None
    """
    # get difference between secondary tensor eigenvalues
    diff = np.abs(vec_tensor_eigen[..., 1] - vec_tensor_eigen[..., 0])

    # get absolute tensor eigenvalues
    avte = np.abs(vec_tensor_eigen)

    # primary dispersion (0.3183098861837907 = 1/π)
    if odi_pri is not None:
        odi_pri[:] = (1 - 0.3183098861837907 * np.arctan2(avte[..., 2], avte[..., 1])).astype(np.float32)

    # secondary dispersion
    if odi_sec is not None:
        odi_sec[:] = (1 - 0.3183098861837907 * np.arctan2(avte[..., 2], avte[..., 0])).astype(np.float32)

    # dispersion anisotropy
    if odi_anis is not None:
        odi_anis[:] = (1 - 0.3183098861837907 * np.arctan2(avte[..., 2], diff)).astype(np.float32)

    # total dispersion
    odi_tot[:] = (1 - 0.3183098861837907 *
                  np.arctan2(avte[..., 2], np.sqrt(np.abs(np.multiply(avte[..., 1], avte[..., 0]))))).astype(np.float32)


def compute_vec_tensor_eigen(fbr_vec):
    """
    Compute the eigenvalues of the 3x3 orientation tensor
    obtained from a reshaped super-voxel of fiber orientation vectors.

    Parameters
    ----------
    fbr_vec: numpy.ndarray (shape=(N,3), dtype=float)
        fiber orientation vectors
        (reshaped super-voxel of shape=(Nz,Ny,Nx), i.e. N=Nz*Ny*Nx)

    Returns
    -------
    vec_tensor_eigen: numpy.ndarray (shape=(3,), dtype=float32)
        orientation tensor eigenvalues in ascending order
    """
    fbr_vec = np.delete(fbr_vec, np.all(fbr_vec == 0, axis=-1), axis=0)
    fbr_vec.shape = (-1, 3)
    t = np.zeros((3, 3))
    for v in fbr_vec:
        t += np.outer(v, v)

    vec_tensor_eigen = sp.linalg.eigh(t, eigvals_only=True).astype(np.float32)

    return vec_tensor_eigen


def generate_odf_background(bg_mrtrix, fbr_vec, vxl_side, iso_fbr=None):
    """
    Generate the downsampled background image required
    to visualize the 3D ODF map in MRtrix3.

    Parameters
    ----------
    bg_mrtrix: NumPy memory-map object (axis order=(X,Y,Z), dtype=uint8)
        initialized background array for ODF visualization in MRtrix3

    fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vectors

    vxl_side: int
        side of the ODF super-voxel [px]

    iso_fbr: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    Returns
    -------
    None
    """
    # select background data
    bg_img = fbr_vec if iso_fbr is None else iso_fbr

    # get shape of new downsampled array
    new_shape = bg_mrtrix.shape[:-1]

    # image normalization: get global minimum and maximum values
    if bg_img.ndim == 3:
        min_glob = np.min(bg_img)
        max_glob = np.max(bg_img)

    # loop over z-slices, and resize them
    for z in range(vxl_side // 2, bg_img.shape[0], vxl_side):
        if bg_img.ndim == 3:
            tmp_slice = normalize_image(bg_img[z], min_val=min_glob, max_val=max_glob)
        elif bg_img.ndim == 4:
            tmp_slice = 255.0 * np.sum(np.abs(bg_img[z, ...]), axis=-1)
            tmp_slice = np.where(tmp_slice <= 255.0, tmp_slice, 255.0)

        tmp_slice = transform_axes(tmp_slice, swapped=(0, 1), flipped=(0, 1))
        bg_mrtrix[..., z // vxl_side] = \
            resize(tmp_slice, output_shape=new_shape, anti_aliasing=True, preserve_range=True)


def init_odf_arrays(vec_img_shp, tmp_dir, odf_scale, odf_deg=6, exp_all=False):
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

    odf_deg: int
        degrees of the spherical harmonics series expansion

    exp_all: bool
        export all images

    Returns
    -------
    odf: NumPy memory-map object (axis order=(X,Y,Z,C), dtype=float32)
                initialized array of ODF spherical harmonics coefficients

    odi: dict
        dispersion image dictionary

            odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                initialized array of primary orientation dispersion parameters

            odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                initialized array of secondary orientation dispersion parameters

            odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                initialized array of total orientation dispersion parameters

            odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                initialized array of orientation dispersion anisotropy parameters

    fbr_dnst: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        initialized fiber density map

    bg_mrtrix: NumPy memory-map object (axis order=(X,Y,Z), dtype=uint8)
        initialized background for ODF visualization in MRtrix3

    vec_tensor_eigen: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        initialized fiber orientation tensor eigenvalues
    """
    # ODI maps shape
    odi_shp = tuple(np.ceil(np.divide(vec_img_shp, odf_scale)).astype(int))

    # create downsampled background memory map
    bg_shp = tuple(np.flip(odi_shp))
    bg_mrtrix = create_memory_map(bg_shp, dtype='uint8', name=f'bg_tmp{odf_scale}', tmp_dir=tmp_dir)

    # create ODF memory map
    nc = get_sph_harm_ncoeff(odf_deg)
    odf_shp = odi_shp + (nc,)
    odf = create_memory_map(odf_shp, dtype='float32', name=f'odf_tmp{odf_scale}', tmp_dir=tmp_dir)

    # create orientation tensor memory map
    vec_tensor_shape = odi_shp + (3,)
    vec_tensor_eigen = create_memory_map(vec_tensor_shape, dtype='float32',
                                         name=f'tensor_tmp{odf_scale}', tmp_dir=tmp_dir)

    # fiber density memory map
    fbr_dnst = create_memory_map(odi_shp, dtype='float32', name=f'fbr_dnst_tmp{odf_scale}', tmp_dir=tmp_dir)

    # create ODI memory maps
    odi_tot = create_memory_map(odi_shp, dtype='float32', name=f'odi_tot_tmp{odf_scale}', tmp_dir=tmp_dir)
    if exp_all:
        odi_pri = create_memory_map(odi_shp, dtype='float32', name=f'odi_pri_tmp{odf_scale}', tmp_dir=tmp_dir)
        odi_sec = create_memory_map(odi_shp, dtype='float32', name=f'odi_sec_tmp{odf_scale}', tmp_dir=tmp_dir)
        odi_anis = create_memory_map(odi_shp, dtype='float32', name=f'odi_anis_tmp{odf_scale}', tmp_dir=tmp_dir)
    else:
        odi_pri, odi_sec, odi_anis = (None, None, None)

    # fill output image dictionary
    odi = dict()
    odi['odi_pri'] = odi_pri
    odi['odi_sec'] = odi_sec
    odi['odi_tot'] = odi_tot
    odi['odi_anis'] = odi_anis

    return odf, odi, fbr_dnst, bg_mrtrix, vec_tensor_eigen


def mask_orientation_dispersion(vec_tensor_eigen, odi):
    """
    Suppress orientation dispersion background.

    Parameters
    ----------
    vec_tensor_eigen: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        orientation tensor eigenvalues
        computed from an ODF super-voxel

    odi: dict
        orientation dispersion dictionary

        odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
            array of primary orientation dispersion parameters

        odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
            array of secondary orientation dispersion parameters

        odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
            array of total orientation dispersion parameters

        odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
            array of orientation dispersion anisotropy parameters

    Returns
    -------
    None
    """
    # mask background
    bg_msk = np.all(vec_tensor_eigen == 0, axis=-1)
    for key in odi.keys():
        if odi[key] is not None:
            odi[key][bg_msk] = 0
