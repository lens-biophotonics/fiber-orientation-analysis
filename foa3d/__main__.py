from foa3d.input import get_cli_parser, load_microscopy_image
from foa3d.pipeline import parallel_odf_over_scales, parallel_frangi_over_slices
from foa3d.printing import print_pipeline_heading
from foa3d.utils import delete_tmp_folder


def foa3d(cli_args):

    # load 3D grayscale or RGB microscopy image or 4D array of fiber orientation vectors
    in_img, save_dirs = load_microscopy_image(cli_args)

    # parallel 3D Frangi-based fiber orientation analysis on batches of basic image slices
    out_img = parallel_frangi_over_slices(cli_args, save_dirs, in_img)

    # generate 3D fiber ODF maps over the spatial scales of interest using concurrent workers
    parallel_odf_over_scales(cli_args, save_dirs, out_img['vec'], out_img['iso'], out_img['px_sz'], in_img['name'])

    # delete temporary folder
    delete_tmp_folder(save_dirs['tmp'], (in_img, out_img))


def main():
    # start Foa3D by terminal
    print_pipeline_heading()
    foa3d(cli_args=get_cli_parser())


if __name__ == '__main__':
    main()
