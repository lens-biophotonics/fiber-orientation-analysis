from os import mkdir, path, scandir
from shutil import copy

import numpy as np
import tifffile as tiff
from skimage.transform import resize

from fiber_orientation.utils import clear_from_memory


class CIDRE:
    ZERO_PRESERVED = 0
    RANGE_CORRECTED = 1
    DIRECT = 2


def print_cidre_heading(cidre_mode):
    """
    Print CIDRE-based illumination correction heading.

    Parameters
    ----------
    cidre_mode: int
        correction mode flag

    Returns
    -------
    None
    """
    hdr_str = '  CIDRE illumination correction\n'
    if cidre_mode == CIDRE.ZERO_PRESERVED:
        hdr_str += '  Mode:       zero-light preserved\n'
    elif cidre_mode == CIDRE.RANGE_CORRECTED:
        hdr_str += '  Mode:       dynamic range corrected\n'
    elif cidre_mode == CIDRE.DIRECT:
        hdr_str += '  Mode:       direct\n'
    hdr_str += '  Objective:  Zeiss\n  Emission \u03BB: 618nm (red)\n              482nm (green)\n'               
    print(hdr_str)


def load_cidre_models(cidre_path, model_shape=(512, 512, 3), channels=['red', 'green', 'blue']):
    """
    Load the spatial models of the gain (v) and additive noise (z) terms of the
    input microscopy images, identified via the CIDRE method
    (red and green channels).

    Parameters
    ----------
    cidre_path:
        path to the CIDRE correction models

    model_shape: tuple
        model array shape (default: (512, 512, 3))

    channels: list
        name of the channel sub-folders

    Returns
    -------
    v: ndarray (shape=(512,512), dtype=float)
        spatial gain function

    z: ndarray (shape=(512,512), dtype=float)
        spatial additive noise
    """
    # initialize empty offset and gain model arrays
    v = np.empty(model_shape)
    v[:] = np.nan
    z = v.copy()
    print('  Channels:   ', end='')

    # load CIDRE models (gain and additive terms, v and z)
    corr_channels = ''
    for c in range(len(channels)):
        v_path = path.join(cidre_path, channels[c], 'v.txt')
        z_path = path.join(cidre_path, channels[c], 'z.txt')
        try:
            v[..., c] = np.loadtxt(v_path, delimiter='\t')
            z[..., c] = np.loadtxt(z_path, delimiter='\t')
        except Exception:
            v[..., c] = np.ones(model_shape[:-1])
            z[..., c] = np.zeros(model_shape[:-1])
            corr_channels += '(skipping '+channels[c]+': models not found)'
        else:
            corr_channels += channels[c]+' '
    print(corr_channels+'\n')

    return v, z


def resize_cidre_models(v, z, slice_shape):
    """
    Resize the CIDRE illumination models if needed.

    Parameters
    ----------
    v: ndarray (shape=(512,512), dtype=float)
        spatial gain function

    z: ndarray (shape=(512,512), dtype=float)
        spatial additive noise

    slice_shape: tuple
        lateral (in-plane) shape of input image stacks

    Returns
    -------
    v_res: ndarray (dtype=float)
        resized spatial gain function

    z_res: ndarray (dtype=float)
        resized spatial additive noise
    """
    # if shapes do not match...
    model_shape = v.shape
    if slice_shape != model_shape:
        print(slice_shape)
        v_res = np.zeros(slice_shape)
        z_res = v_res.copy()

        # resize models, looping over channels
        for c in range(slice_shape[-1]):
            v_res[..., c] = resize(v[..., c], slice_shape[:-1], anti_aliasing=True, preserve_range=True)                                  
            z_res[..., c] = resize(z[..., c], slice_shape[:-1], anti_aliasing=True, preserve_range=True)

        return v_res, z_res

    else:
        return v, z


def apply_cidre_models(slice_rgb, v, z, v_mean, z_mean, cidre_mode=CIDRE.ZERO_PRESERVED, pro_type=np.float64):
    """
    Apply CIDRE illumination correction to 2D RGB slice.

    Parameters
    ----------
    slice_rgb: ndarray (shape=(Y,X,C))
        RGB slice

    v: ndarray (shape=(Y,X,C), dtype=float)
        gain model

    z: ndarray (shape=(Y,X,C), dtype=float)
        offset model

    v_mean: float
        mean gain value (computed once for efficiency's sake)

    z_mean: float
        mean offset value (computed once for efficiency's sake)

    cidre_mode: int
        correction mode flag

    pro_type:
        data processing type

    Returns
    -------
    corr_slice: ndarray (shape=(Y,X,C))
        corrected RGB slice
    """
    # original conversion
    slice_rgb = slice_rgb.astype(pro_type)

    # initialize zero slice
    corr_slice = np.zeros_like(slice_rgb)

    # zero-light preserved
    if cidre_mode == CIDRE.ZERO_PRESERVED:
        for c in range(3):
            if v_mean[c] != np.nan:
                corr_slice[..., c] = z_mean[c] + v_mean[c] * (np.divide(slice_rgb[..., c] - z[..., c], v[..., c]))

    # dynamic range corrected
    elif cidre_mode == CIDRE.RANGE_CORRECTED:
        for c in range(3):
            if v_mean[c] != np.nan:
                corr_slice[..., c] = v_mean[c] * (np.divide(slice_rgb[..., c] - z[..., c], v[..., c]))

    # direct correction
    elif cidre_mode == CIDRE.DIRECT:
        for c in range(3):
            if v_mean[c] != np.nan:
                corr_slice[..., c] = np.divide(slice_rgb[..., c] - z[..., c], v[..., c])

    return corr_slice


def correct_illumination(source, models='/mnt/NASone/michele/fiberSor/cidre', mosaic=False, mode=CIDRE.ZERO_PRESERVED):
    """
    Correct the uneven illumination of the input TPFM image stacks,
    using the CIDRE illumination-correction method (Smith et al. 2014).

    Parameters
    ----------
    source: str
        source path string
        (single stack file or directory including multiple stacks)

    models: str
        path to directory including the identified models of illumination gain
        (v) offset (z) used for correction\n
        (they must be placed inside sub-folders named after the related channel
        to be corrected, i.e. ../cidre/green/v.txt, ../cidre/red/z.txt;
        channels not associated to any models won't be corrected)

    mosaic: bool
        must be True for tiled reconstructions aligned using ZetaStitcher

    cidre_mode: int
        correction mode flag

    Returns
    -------
    corr_source: path object
        path to corrected source file/s
    """
    # print CIDRE correction heading
    print_cidre_heading(mode)

    # create sub-folder where storing the original stacks
    zstack_dir = path.dirname(source)
    cidre_out_dir = path.join(zstack_dir, 'cidre_corrected_mode'+str(mode))

    # get new CIDRE-corrected source path
    corr_source = path.join(cidre_out_dir, path.basename(source))

    # apply correction if not already performed
    if not path.isfile(corr_source):
        if not path.isdir(cidre_out_dir):
            mkdir(cidre_out_dir)

        # load CIDRE spatial models
        # (gain and offset terms, v and z)
        v, z = load_cidre_models(models)
        resized = False

        # compute mean values
        v_mean = np.mean(v, axis=(0, 1))
        z_mean = np.mean(z, axis=(0, 1))

        # single TIF stack
        if not mosaic:
            stacks = [source]

        # multiple TIF stacks
        else:

            # get all z-stack filenames inside the source folder
            obj = scandir(zstack_dir)
            stacks = list()
            for entry in obj:
                if entry.is_file() and entry.name.endswith('tiff'):
                    stacks.append(path.join(zstack_dir, entry.name))

        # loop over the z-stack filenames
        Ns = len(stacks)
        loop_count = 1
        for s in stacks:

            # print progress
            prc_progress = 100 * (loop_count / Ns)
            print('  correcting stack {0}/{1}: {2:0.1f}%'.format(loop_count, Ns, prc_progress), end='\r')                  

            # load z-stack
            volume_in = tiff.imread(s)
            volume_out = np.zeros_like(volume_in)
            volume_shape = volume_in.shape
            volume_dtype = volume_in.dtype
            max_out = np.iinfo(volume_dtype).max
            if not resized:
                v, z = resize_cidre_models(v, z, slice_shape=volume_shape[1:])
                resized = True

            # loop over z-slices
            for d in range(volume_shape[0]):

                # apply CIDRE correction
                volume_out[d, ...] = \
                    apply_cidre_models(volume_in[d, ...], v, z, v_mean=v_mean, z_mean=z_mean,
                                       cidre_mode=mode, pro_type=np.float64)                                      

            # clip values
            volume_out = np.where(volume_out >= 0, volume_out, 0)
            volume_out = np.where(volume_out <= max_out, volume_out, max_out)

            # convert back to the original data type
            volume_out = volume_out.astype(volume_dtype)

            # overwrite volume
            tiff.imwrite(path.join(cidre_out_dir, path.basename(s)), volume_out)                         

            # clear input stack from memory
            clear_from_memory(volume_in)

            # increase loop counter
            loop_count += 1

        # skip line
        print('\n')

    # copy .yml stitch file to the CIDRE sub-folder
    if mosaic and not path.exists(corr_source):
        copy(source, corr_source)

    return corr_source
