from modules.input import (cli_parser_config, load_pipeline_config,
                           load_input_volume)
from modules.pipeline import (iterate_frangi_on_slices, iterate_odf_on_slices,
                              save_frangi_volumes, save_odf_volumes)
from modules.printing import print_odf_heading, print_pipeline_heading
from modules.utils import delete_tmp_files


def fiberSor(cli_parser):

    # load image volume or dataset of fiber orientation vectors
    volume, mosaic, skip_frangi \
        = load_input_volume(cli_parser)

    # get pipeline configuration
    alpha, beta, gamma, scales_um, smooth_sigma, px_size, px_size_iso, \
        odf_scales_um, odf_orders, z_min, z_max, ch_neuron, ch_fiber, \
        max_slice_size, lpf_soma_mask, save_dir, volume_name \
        = load_pipeline_config(cli_parser)

    # iteratively apply 3D Frangi-based fiber enhancement to basic image slices
    if not skip_frangi:
        tmp_hdf5_list, vec_volume, vec_colmap, frangi_volume, \
            iso_fiber_volume, fiber_mask, neuron_mask \
            = iterate_frangi_on_slices(volume, px_size, px_size_iso,
                                       smooth_sigma, save_dir, volume_name,
                                       max_slice_size=max_slice_size,
                                       alpha=alpha, beta=beta, gamma=gamma,
                                       z_min=z_min, z_max=z_max,
                                       scales_um=scales_um,
                                       lpf_soma_mask=lpf_soma_mask,
                                       ch_neuron=ch_neuron, ch_fiber=ch_fiber,
                                       mosaic=mosaic)

        # save Frangi filtering volumes to TIF files
        save_frangi_volumes(vec_volume, vec_colmap, frangi_volume,
                            fiber_mask, neuron_mask, save_dir, volume_name)

    # estimate 3D fiber ODF maps iterating over the input list of ODF scales
    if odf_scales_um:
        odf_volume_list = []
        bg_mrtrix_volume_list = []

        # fiber vectors provided as input
        if skip_frangi:
            tmp_hdf5_list = []
            vec_volume = volume
            iso_fiber_volume = None

        # print ODF analysis heading
        print_odf_heading(odf_scales_um, odf_orders)
        for odf_scale_um in odf_scales_um:
            odf_volume, bg_mrtrix_volume, tmp_hdf5_list \
                = iterate_odf_on_slices(vec_volume, iso_fiber_volume,
                                        px_size_iso, save_dir,
                                        max_slice_size=max_slice_size,
                                        tmp_files=tmp_hdf5_list,
                                        odf_scale_um=odf_scale_um,
                                        odf_orders=odf_orders)
            odf_volume_list.append(odf_volume)
            bg_mrtrix_volume_list.append(bg_mrtrix_volume)

        # save ODF analysis volumes to TIF and Nifti files (Mrtrix3)
        save_odf_volumes(odf_volume_list, bg_mrtrix_volume_list,
                         save_dir, volume_name, odf_scales_um=odf_scales_um)

    # remove temporary HDF5 files
    delete_tmp_files(tmp_hdf5_list)


if __name__ == '__main__':

    # start fiberSor pipeline by terminal
    print_pipeline_heading()
    fiberSor(cli_parser=cli_parser_config())
