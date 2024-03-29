import argparse
import numpy as np
import cv2
from PIL import Image
import os

from utils.config import Config
from utils.validation_utils import predict_on_img
from models.detection_models import get_model

parser = argparse.ArgumentParser(description='Script for inference on big image.')
parser.add_argument('model_name', type=str, choices=['unet', 'attunet'],
                    help='Name of the model between "unet" and "attunet" ')
parser.add_argument('checkpoint', type=str,
                    help='Path to the .hdf5 checkpoint file')
parser.add_argument('img_path', type=str,
                    help='Path pointing to the image to inference on')
parser.add_argument('depth_mask_path', type=str,
                    help='Path pointing to the realtive depth mask. Leave "" in order to not use it')
parser.add_argument('output_path', type=str,
                    help='Where to store the predictions')
parser.add_argument('crop_size', type=int,
                    help='Size of the crops')
parser.add_argument('scale_factor', type=int,
                    help='Use same scale factor as in training')
parser.add_argument('--use_dsm', action=argparse.BooleanOptionalAction,
                    help='Whether preprocess also LIDAR data')
def inference():
    args = parser.parse_args()
    config = Config(
        crop_size=args.crop_size,
        use_dsm= args.use_dsm,
        model_name = args.model_name,
        checkpoint=args.checkpoint,
        step= int(args.crop_size)/2,
        scale_factor= args.scale_factor
    )
    print(config.num_channels)
    model = get_model(config)
    pred_rescaled = predict_on_img(model,
                                    img_path= args.img_path,
                                    depth_mask_path= args.depth_mask_path if args.use_dsm else "",
                                    batch_size=config.batch_size,
                                    crop_size= config.crop_size,
                                    step= config.step,
                                    scale_factor= config.scale_factor,
                                    )
    im = np.array(Image.open(args.img_path))
    pred= cv2.resize(pred_rescaled, (im.shape[1], im.shape[0]))
    pred_save = pred * 255
    pred_binary = (pred > 0.5) * 255
    img_basename = os.path.basename(args.img_path).split(".")[0]
    cv2.imwrite(os.path.join(args.output_path,f'{img_basename}_pred.png'), pred_save)
    cv2.imwrite(os.path.join(args.output_path,f'{img_basename}_pred_binary.png'), pred_binary)

if __name__ == '__main__':
    inference()