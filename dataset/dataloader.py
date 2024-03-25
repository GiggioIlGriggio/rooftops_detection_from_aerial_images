from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def my_image_mask_generator(image_data_generator, mask_data_generator, depth_data_generator = None):
    if depth_data_generator:
      train_generator = zip(image_data_generator, mask_data_generator, depth_data_generator)
      for (img, mask, depth) in train_generator:
          x = np.concatenate((img, depth), axis=-1)
          yield (x, mask)
    else:
      train_generator = zip(image_data_generator, mask_data_generator)
      for (img, mask) in train_generator:
          yield (img, mask)
def preprocess_mask(image):
    binary_mask = np.where(image > 0, 1, 0)
    return binary_mask


def get_dataloaders(config):
    crop_size, seed, batch_size = config.crop_size, config.seed, config.batch_size

    train_data_generator = ImageDataGenerator(rescale=1./255, horizontal_flip = True, vertical_flip = True)  # You can add other augmentation parameters here
    train_data_generator_mask = ImageDataGenerator(dtype= np.float16, horizontal_flip = True, vertical_flip = True)# preprocessing_function=preprocess_mask
    train_data_generator_depth = ImageDataGenerator(rescale=1./255, dtype= np.float16, horizontal_flip = True, vertical_flip = True)# preprocessing_function=preprocess_mask
    val_data_generator = ImageDataGenerator(rescale=1./255) #dtype= np.uint8) #rescale=1./255)  # You can add other augmentation parameters here
    val_data_generator_mask = ImageDataGenerator(dtype= np.float16) # np.uint8) #np.float16)# preprocessing_function=preprocess_mask
    val_data_generator_depth = ImageDataGenerator(rescale=1./255, dtype= np.float16) # np.uint8) #np.float16)# preprocessing_function=preprocess_mask

    # Define paths to your data/content/drive/MyDrive/cropped_dataset_4_256_128_TEST
    train_data_dir = os.path.join(config.working_dir, "datasets", config.dataset_name, "train")
    val_data_dir = os.path.join(config.working_dir, "datasets", config.dataset_name, "val")

    # Define batch size

    # Generate batches of data
    train_generator = train_data_generator.flow_from_directory(
        train_data_dir,
        target_size=(crop_size, crop_size),  # Set the target size of your images
        batch_size=batch_size,
        class_mode=None,  # Set class_mode to 'input' to load images as input
        color_mode='rgb',  # Set color_mode to 'rgb' for 3 channels (if your images are RGB)
        shuffle=True,  # Set shuffle to True for training
        classes=['images'],  # Specify the subdirectory containing input images
        seed = seed
    )

    train_mask_generator = train_data_generator_mask.flow_from_directory(
        train_data_dir,
        target_size=(crop_size, crop_size),
        batch_size=batch_size,
        class_mode=None,  # Set class_mode to 'input' to load masks as input
        color_mode='grayscale',  # Set color_mode to 'grayscale' for single channel masks
        shuffle=True,  # Set shuffle to True for training
        classes=['masks'],  # Specify the subdirectory containing masks
        seed = seed
    )
    
    if config.use_dsm:
        train_depth_generator = train_data_generator_depth.flow_from_directory(
            train_data_dir,
            target_size=(crop_size, crop_size),
            batch_size=batch_size,
            class_mode=None,  # Set class_mode to 'input' to load masks as input
            color_mode='grayscale',  # Set color_mode to 'grayscale' for single channel masks
            shuffle=True,  # Set shuffle to True for training
            classes=['depths'],  # Specify the subdirectory containing masks
            seed = seed
        )
    else:
        train_depth_generator = None

    # Combine the generators to yield both images and masks
    train_combined_generator = my_image_mask_generator(train_generator, train_mask_generator, train_depth_generator)

    # Similar setup for validation data
    validation_generator = val_data_generator.flow_from_directory(
        val_data_dir,
        target_size=(crop_size, crop_size),
        batch_size=batch_size,
        class_mode=None,
        color_mode='rgb',
        shuffle=False,  # Set shuffle to False for validation
        classes=['images'],
        seed = seed
    )

    validation_mask_generator = val_data_generator_mask.flow_from_directory(
        val_data_dir,
        target_size=(crop_size, crop_size),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        shuffle=False,  # Set shuffle to False for validation
        classes=['masks'],
        seed = seed
    )

    if config.use_dsm:
        validation_depth_generator = val_data_generator_depth.flow_from_directory(
            val_data_dir,
            target_size=(crop_size, crop_size),
            batch_size=batch_size,
            class_mode=None,
            color_mode='grayscale',
            shuffle=False,  # Set shuffle to False for validation
            classes=['depths'],
            seed = seed
        )
    else:
        validation_combined_generator = None

    validation_combined_generator = my_image_mask_generator(validation_generator, validation_mask_generator, validation_depth_generator)

    return train_combined_generator, validation_combined_generator, train_generator.samples,  validation_generator.samples