import argparse

from utils.config import Config
from utils.dataset_handler import create_dataset


parser = argparse.ArgumentParser(description='Script for storing the preprocessed data to the drive.')
parser.add_argument('working_dir', metavar='working_dir', type=str,
                    help='Path to all the outputs of the model')
parser.add_argument('dataset_name', type=str,
                    help='Folder name of the output dataset')
parser.add_argument('crop_size', type=int,
                    help='Size of the crops')
parser.add_argument('step', type=int,
                    help='Step size')
parser.add_argument('preprocessed_dataset_name', type=str,
                    help='Folder name of the output preprocessed data')
parser.add_argument('scale_factor', type=int,
                    help='Scale factor to rescale the images, suggested to use 4')
parser.add_argument('--use_dsm', action=argparse.BooleanOptionalAction,
                    help='Whether preprocess also LIDAR data')
parser.add_argument('--val_names', nargs='+', required= True,
                    help='List of validation images names without extension')
def main():
    args = parser.parse_args()
    config = Config(
        working_dir = args.working_dir,
        preprocessed_dataset_name= args.preprocessed_dataset_name,
        dataset_name= args.dataset_name,
        crop_size= args.crop_size,
        val_names= args.val_names,
        scale_factor= args.scale_factor,
        step= args.step,
        use_dsm= args.use_dsm
    )

    create_dataset(config)


if __name__ == '__main__':
    main()