import argparse
import os
import glob
import json

from utils.config import Config
from utils.validation_utils import compute_metrics_all_val
from models.detection_models import get_model
from utils.utils import convert_to_serializable



parser = argparse.ArgumentParser(description='Script for inference on big image.')
parser.add_argument('model_name', type=str, choices=['unet', 'attunet'],
                    help='Name of the model between "unet" and "attunet" ')
parser.add_argument('checkpoints_dir', type=str,
                    help='Name of the checkpoint folder (name of the configuration)')
parser.add_argument('working_dir', type=str,
                    help='Path to the working dir')
parser.add_argument('preprocessed_dataset_name', type=str,
                    help='Name of the preprocessed dataset')
parser.add_argument('crop_size', type=int,
                    help='Size of the crops')
parser.add_argument('scale_factor', type=int,
                    help='Use same scale factor as in training')
parser.add_argument('--save_inference', action=argparse.BooleanOptionalAction,
                    help='Whether preprocess also LIDAR data')
parser.add_argument('--use_dsm', action=argparse.BooleanOptionalAction,
                    help='Whether preprocess also LIDAR data')
parser.add_argument('--val_names', nargs='+', required= True,
                    help='List of validation images names without extension')

def inference():
    args = parser.parse_args()
    config = Config(
        working_dir=args.working_dir,
        preprocessed_dataset_name= args.preprocessed_dataset_name,
        crop_size=args.crop_size,
        use_dsm= args.use_dsm,
        model_name = args.model_name,
        checkpoints_dir=args.checkpoints_dir,
        step= int(args.crop_size)/2,
        scale_factor= args.scale_factor,
        val_names= args.val_names,
    )
    print(config.num_channels)
    model = get_model(config)
    checkpoints_dir = os.path.join(config.working_dir,"checkpoints",config.checkpoints_dir)
    if os.listdir(checkpoints_dir):
        best_model_ID = sorted([path[-11:] for path in glob.glob(os.path.join(checkpoints_dir, '*.hdf5'), recursive=False)])[-1]
        best_model_path = [path for path in glob.glob(os.path.join(checkpoints_dir, '*.hdf5'), recursive=False) if best_model_ID in path][0]
        print(f"Loading best model found at path {best_model_path}")
        model.load_weights(best_model_path)
    else:
        print(f"ERROR, no checkpoitns found in the specified folder {checkpoints_dir}")
    metrics_dict = compute_metrics_all_val(model,config, save_inference=args.save_inference)
    
    output_json = os.path.join(config.working_dir,"checkpoints",config.checkpoints_dir, "evaluation.json")
    with open(output_json, "w") as json_file:
        json.dump(metrics_dict, json_file, default= convert_to_serializable)
    print(f"All results saved at {output_json}")

if __name__ == '__main__':
    inference()