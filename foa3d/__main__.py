from foa3d.input import cli_parser, load_microscopy_image, get_pipeline_config
from foa3d.pipeline import (iterate_frangi_on_slices, iterate_odf_on_slices,
                            save_frangi_arrays, save_odf_arrays)
from foa3d.printing import print_odf_heading, print_pipeline_heading
from foa3d.utils import delete_tmp_files


def foa3d(cli_args):

    # load image volume or dataset of fiber orientation vectors
    image, mosaic, skip_frangi, cli_args = load_microscopy_image(cli_args)

    # get pipeline configuration
    alpha, beta, gamma, scales_um, smooth_sigma, px_size, px_size_iso, odf_scales_um, odf_degrees, z_min, z_max, \
        ch_neuron, ch_fiber, max_slice_size, lpf_soma_mask, save_dir, image_name = get_pipeline_config(cli_args)

    # iteratively apply 3D Frangi-based fiber enhancement to basic image slices
    if not skip_frangi:
        fiber_vec_image, fiber_vec_colmap, frangi_image, \
            iso_fiber_image, fiber_mask, neuron_mask, tmp_file_lst \
            = iterate_frangi_on_slices(image, px_size, px_size_iso, smooth_sigma, save_dir, image_name,
                                       max_slice_size=max_slice_size, alpha=alpha, beta=beta, gamma=gamma,
                                       z_min=z_min, z_max=z_max, scales_um=scales_um, lpf_soma_mask=lpf_soma_mask,
                                       ch_neuron=ch_neuron, ch_fiber=ch_fiber, mosaic=mosaic)

        # save Frangi filtering arrays to TIF files
        save_frangi_arrays(fiber_vec_colmap, frangi_image, fiber_mask, neuron_mask, px_size_iso, save_dir, image_name)

    # estimate 3D fiber ODF maps iterating over the input list of ODF scales
    if odf_scales_um:
        odf_image_lst = []
        bg_mrtrix_image_lst = []

        # fiber vectors provided as input
        if skip_frangi:
            tmp_file_lst = []
            fiber_vec_image = image
            iso_fiber_image = None

        # print ODF analysis heading
        print_odf_heading(odf_scales_um, odf_degrees)
        for odf_scale_um in odf_scales_um:
            odf_image, bg_mrtrix_image, tmp_file_lst \
                = iterate_odf_on_slices(fiber_vec_image, iso_fiber_image, px_size_iso, save_dir,
                                        max_slice_size=max_slice_size, tmp_file_lst=tmp_file_lst,
                                        odf_scale_um=odf_scale_um, odf_degrees=odf_degrees)
            odf_image_lst.append(odf_image)
            bg_mrtrix_image_lst.append(bg_mrtrix_image)

        # save ODF analysis arrays to TIF and Nifti files (Mrtrix3)
        save_odf_arrays(odf_image_lst, bg_mrtrix_image_lst, save_dir, image_name, odf_scales_um=odf_scales_um)

    # remove temporary HDF5 files
    delete_tmp_files(tmp_file_lst)


def main():
    # start Foa3D pipeline by terminal
    print_pipeline_heading()
    foa3d(cli_args=cli_parser())


if __name__ == '__main__':
    main()
