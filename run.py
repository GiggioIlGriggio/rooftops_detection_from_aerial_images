import argparse
from models.detection_models import get_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import glob 

from models.callbacks import DisplayCallback
from dataset.dataloader import get_dataloaders
from utils.config import Config
from models.losses import get_loss

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(description='Script for preprocessing the data or store the preprocessed data to the drive.')
parser.add_argument('working_dir', metavar='working_dir', type=str,
                    help='Path to all the outputs of the model')
parser.add_argument('model_name', type=str, choices=['unet', 'attunet'],
                    help='Name of the model between "unet" and "attunet" ')
parser.add_argument('dataset_name', type=str,
                    help='Folder name of the output dataset')
parser.add_argument('crop_size', type=int,
                    help='Size of the crops')
parser.add_argument('--use_dsm', action=argparse.BooleanOptionalAction,
                    help='Whether preprocess also LIDAR data')
parser.add_argument('batch_size', type=int,
                    help='Number of images per batch')
parser.add_argument('num_epochs', type=int,
                    help='Number of epochs')
parser.add_argument('loss', type=str,
                    help='String containing the losses to use separeted by a "_". They can be "tversky", "iou" or "binary". Ex "binary_iou"')
parser.add_argument('checkpoints_dir', type=str,
                    help='Folder name of the output checkpoint weights')


def run_training(config):
    train_generator, val_generator, train_samples, val_samples = get_dataloaders(config)

    model = get_model(config)

    reduce_lr_callback = ReduceLROnPlateau(monitor='loss', patience=4, min_delta=0.01)
    
    checkpoints_folder = os.path.join(config.working_dir, "checkpoints")
    os.makedirs(checkpoints_folder, exist_ok= True)
    checkpoint_config_folder = os.path.join(checkpoints_folder, config.checkpoints_dir)
    os.makedirs(checkpoint_config_folder, exist_ok= True)
    callback_checkpoint = ModelCheckpoint(os.path.join(checkpoint_config_folder,f'{config.model_name}'+'.{epoch:02d}-{val_binary_accuracy:.4f}.hdf5'), save_weights_only=True,
                                        save_best_only=True, monitor='val_binary_accuracy')
    # TODO handle unet3plus call checkpoint name
    
    # Visualize evolution of the predictions on the 5th validation crop during training
    index = 5
    dataset_path = os.path.join(config.working_dir, "datasets", config.dataset_name, "val")
    crop_name = os.path.basename(os.listdir(os.path.join(dataset_path,"images"))[index])
    callback_im = np.array(Image.open(os.path.join(dataset_path, "images", crop_name)))
    callback_mask = np.array(Image.open(os.path.join(dataset_path, "masks", crop_name)))

    if config.use_dsm:
        callback_depth = np.array(Image.open(os.path.join(dataset_path, "depths", crop_name)))
        callback_depth = np.expand_dims(callback_depth, axis= -1)
        callback_im = np.concatenate((callback_im, callback_depth), axis=-1)
    callback_im = callback_im.astype(np.float32) / 255
    display_callback = DisplayCallback(callback_im, callback_mask, 5)

    callbacks = [display_callback, callback_checkpoint] #reduce_lr_callback

    train_steps_per_epoch = train_samples // config.batch_size
    validation_steps_per_epoch = val_samples // config.batch_size

    num_epochs_1e3 = config.num_epochs//3 * 2
    num_epochs_1e4 = config.num_epochs//6
    num_epochs_1e5 = config.num_epochs//6

    checkpoints_dir = os.path.join(config.working_dir,"checkpoints",config.checkpoints_dir)
    if os.listdir(checkpoints_dir):
        best_model_path = sorted(glob.glob(os.path.join(checkpoints_dir, '*.hdf5'), recursive=False))
        print(f"Loading best model found at path {best_model_path}")
        model.load_weights(best_model_path)
    else:
        print("Training model from scratch")
    
    print("Training with learning rate 1e-3")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=get_loss(config.loss),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

    hist = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=num_epochs_1e3,
        validation_data=val_generator,
        validation_steps=validation_steps_per_epoch,
        callbacks=callbacks
    )

    print("Training with learning rate 1e-4")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=get_loss(config.loss),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    best_model_path = sorted(glob.glob(os.path.join(checkpoints_dir, '*.hdf5'), recursive=False))
    print(f"Loading best model found at path {best_model_path}")
    model.load_weights(best_model_path)

    hist = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=num_epochs_1e4,
        validation_data=val_generator,
        validation_steps=validation_steps_per_epoch,
        callbacks=callbacks
    )

    print("Training with learning rate 1e-5")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=get_loss(config.loss),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

    best_model_path = sorted(glob.glob(os.path.join(checkpoints_dir, '*.hdf5'), recursive=False))
    print(f"Loading best model found at path {best_model_path}")
    model.load_weights(best_model_path)
    
    hist = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=num_epochs_1e5,
        validation_data=val_generator,
        validation_steps=validation_steps_per_epoch,
        callbacks=callbacks
    )

def main():
    args = parser.parse_args()
    config = Config(
        working_dir = args.working_dir,
        model_name= args.model_name,
        dataset_name= args.dataset_name,
        crop_size= args.crop_size,
        use_dsm= args.use_dsm,
        batch_size= args.batch_size,
        num_epochs= args.num_epochs,
        loss= args.loss,
        checkpoints_dir= args.checkpoints_dir
    )
    run_training(config)

if __name__ == '__main__':
    main()