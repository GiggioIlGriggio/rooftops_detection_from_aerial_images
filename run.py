from models.detection_models import get_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import numpy as np
from PIL import Image

from models.callbacks import DisplayCallback
from dataset.dataloader import get_dataloaders

def run_training(config):
    train_generator, val_generator, train_samples, val_samples = get_dataloaders(config)

    model = get_model(config)

    reduce_lr_callback = ReduceLROnPlateau(monitor='loss', patience=4, min_delta=0.01)
    
    checkpoints_folder = os.path.join(config.working_dir, "checkpoints")
    os.makedirs(checkpoints_folder, exist_ok= True)
    checkpoint_config_folder = os.poath.join(checkpoints_folder, config.config_name)
    os.makedirs(checkpoint_config_folder, exist_ok= True)
    callback_checkpoint = ModelCheckpoint(os.path.join(checkpoint_config_folder,'{config.model_name}.{epoch:02d}-{val_binary_accuracy:.4f}.hdf5'), save_weights_only=True,
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

    display_callback = DisplayCallback(callback_im, callback_mask, 5)

    callbacks = [reduce_lr_callback, callback_checkpoint, display_callback]

    train_steps_per_epoch = train_samples // config.batch_size
    validation_steps_per_epoch = val_samples // config.batch_size
    hist = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=config.num_epochs,
        validation_data=val_generator,
        validation_steps=validation_steps_per_epoch,
        callbacks=callbacks,
        #workers=2,
        #use_multiprocessing=True
    )
