from foa3d.input import get_cli_parser, get_resource_config, load_microscopy_image
from foa3d.pipeline import (parallel_odf_at_scales, parallel_frangi_on_slices)
from foa3d.printing import print_pipeline_heading
from foa3d.utils import delete_tmp_folder


def foa3d(cli_args):

    # load 3D microscopy image or 4D array of fiber orientation vectors
    img, ts_msk, is_tiled, is_fiber, save_dir, tmp_dir, img_name = load_microscopy_image(cli_args)

    # get resources configuration
    ram, jobs = get_resource_config(cli_args)

    # conduct parallel 3D Frangi-based fiber orientation analysis on batches of basic image slices
    if not is_fiber:
        fbr_vec_img, iso_fbr_img, px_sz, img_name \
            = parallel_frangi_on_slices(img, cli_args, save_dir[0], tmp_dir, img_name,
                                        ts_msk=ts_msk, ram=ram, jobs=jobs, is_tiled=is_tiled)
    else:
        fbr_vec_img, iso_fbr_img, px_sz = (img, None, None)

    # estimate 3D fiber ODF maps at the spatial scales of interest using concurrent workers
    if cli_args.odf_res:
        parallel_odf_at_scales(fbr_vec_img, iso_fbr_img, cli_args, px_sz, save_dir[1], tmp_dir, img_name, ram=ram)

    # delete temporary folder
    delete_tmp_folder(tmp_dir)


def main():
    # start Foa3D by terminal
    print_pipeline_heading()
    foa3d(cli_args=get_cli_parser())


if __name__ == '__main__':
    main()
