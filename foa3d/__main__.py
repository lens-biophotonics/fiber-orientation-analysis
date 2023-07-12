from foa3d.input import get_cli_parser, get_pipeline_config, load_microscopy_image
from foa3d.pipeline import (parallel_odf_on_scales, parallel_frangi_on_slices)
from foa3d.printing import print_pipeline_heading
from foa3d.utils import delete_tmp_folder


def foa3d(cli_args):

    # load microscopy volume image or array of fiber orientation vectors
    img, mosaic, skip_frangi, cli_args, save_subdirs, tmp_dir, img_name = load_microscopy_image(cli_args)

    # get the fiber orientation analysis pipeline configuration
    alpha, beta, gamma, scales_um, smooth_sigma, px_size, px_size_iso, \
        odf_scales_um, odf_degrees, z_min, z_max, ch_neuron, ch_fiber, \
        lpf_soma_mask, max_ram_mb, jobs, img_name = get_pipeline_config(cli_args, skip_frangi, img_name)

    # conduct parallel 3D Frangi-based fiber orientation analysis on batches of basic image slices
    if not skip_frangi:
        fiber_vec_img, iso_fiber_img \
            = parallel_frangi_on_slices(img, px_size, px_size_iso, smooth_sigma, save_subdirs[0], tmp_dir, img_name,
                                        alpha=alpha, beta=beta, gamma=gamma, frangi_sigma_um=scales_um,
                                        z_min=z_min, z_max=z_max, lpf_soma_mask=lpf_soma_mask,
                                        ch_neuron=ch_neuron, ch_fiber=ch_fiber, mosaic=mosaic,
                                        max_ram_mb=max_ram_mb, jobs=jobs)

    # estimate 3D fiber ODF maps over the spatial scales of interest using concurrent workers
    if odf_scales_um:
        if skip_frangi:
            fiber_vec_img = img
            iso_fiber_img = None
        parallel_odf_on_scales(fiber_vec_img, iso_fiber_img, px_size_iso, save_subdirs[1], tmp_dir, img_name,
                               odf_scales_um=odf_scales_um, odf_degrees=odf_degrees, max_ram_mb=max_ram_mb)

    # delete temporary folder
    delete_tmp_folder(tmp_dir)


def main():
    # start Foa3D pipeline by terminal
    print_pipeline_heading()
    foa3d(cli_args=get_cli_parser())


if __name__ == '__main__':
    main()
