import argparse
import glob
import os

from utils.config import Config, INTERNALS_DICT
from utils.dataset_handler import preprocess_data


parser = argparse.ArgumentParser(description='Script for preprocessing the data.')
parser.add_argument('working_dir', metavar='working_dir', type=str,
                    help='Path to all the outputs of the model')
parser.add_argument('preprocessed_dataset_name', type=str,
                    help='Folder name of the output preprocessed data')
parser.add_argument('images_folder', type=str,
                    help='Path to the folder where images to be processed are stored')
parser.add_argument('internals', type=str,
                    help='Name of the internal parameters')
parser.add_argument('--use_dsm', action=argparse.BooleanOptionalAction,
                    help='Whether preprocess also LIDAR data')
parser.add_argument('dsm_path', type=str,
                    help='Path to the folder containing all the LIDAR data. If LIDAR data is not avaible leave empty.')
parser.add_argument('dbfs_path', type=str,
                    help='Path to the folder of dbf files containing external paramters')
parser.add_argument('polylines_path', type=str,
                    help='Path to the dxf file containing rooftop polygons')
parser.add_argument('buffer_path', type=str,
                    help='Path to the dxf file containing buffer polygon')
parser.add_argument('--depth_interpolation', action=argparse.BooleanOptionalAction,
                    help='Whether to interpolate the sparse matrix given by the LIDAR data')


def main():
    args = parser.parse_args()
    config = Config(
        working_dir = args.working_dir,
        preprocessed_dataset_name= args.preprocessed_dataset_name,
        images_paths= glob.glob(os.path.join(args.images_folder, '*.tif'), recursive=True),
        internals= INTERNALS_DICT[args.internals],
        use_dsm= args.use_dsm,
        dsm_path= args.dsm_path,
        dbfs_path= args.dbfs_path,
        polylines_path= args.polylines_path,
        buffer_path = args.buffer_path,
        depth_interpolation= args.depth_interpolation
    )

    preprocess_data(config)


if __name__ == '__main__':
    main()