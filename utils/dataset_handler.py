from utils.utils import Camera, Dxf, AerialPicture, preprocess_mask_image, get_crop_index
from utils.config import Config


import os
import numpy as np
from PIL import Image
import cv2
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None

def preprocess_data(config):
    """
      The function takes the input files, such as the images and the related camera parameters, 
      the dxf file containing the rooftops polylines, the dxf file identifying the raster and
      the tif files relative to the LIDAR data.

      The function calls all the projections functions to produce the output.
      The preprocessing consists in 
        -matching images, camera parameters, rooftop polylines and LIDAR data if avaible
        -cutting the images, masks and LIDAR data to the region in which ground truth data
         is avaible. Since the depth mask given from the LIDAR data has less resolution, it 
         can be interpolated.  

      The output is stored in working_dir/preprocessed_data/config.preprocessed_dataset_name
      and it consist of 3 folders:
        imgs: contains the original images
        masks: contains the rooftops ground truth masks
        depths: contains the masks with LIDAR data

    """
    if config.use_dsm and config.dsm_path == "":
       print("ERROR. dsm_path must be provided if you want to preprocess LIDAR data!")
       return

    print("Creating dirs...")
    preprocessed_data_path = os.path.join(config.working_dir, "preprocessed_data")
    os.makedirs(preprocessed_data_path, exist_ok=True)
    output_path = os.path.join(preprocessed_data_path, config.preprocessed_dataset_name)
    os.makedirs(output_path, exist_ok=True)

    masks_path = os.path.join(output_path, "masks")
    os.makedirs(masks_path, exist_ok=True)
    depths_path = os.path.join(output_path, "depths")
    os.makedirs(depths_path, exist_ok=True)
    imgs_path = os.path.join(output_path, "imgs")
    os.makedirs(imgs_path, exist_ok=True)
    
    #Compute only images that have never been processed
    old_imgs_list = [path.split(".")[0] for path in os.listdir(os.path.join(config.working_dir, "preprocessed_data", config.preprocessed_dataset_name,"imgs"))]
    imgs_to_process_paths = [path for path in config.images_paths if os.path.basename(path).split(".")[0] not in old_imgs_list]
    if not imgs_to_process_paths:
       print(f"All images provided are already preprocessed in {output_path}")
       return
    if imgs_to_process_paths:
        print("New images to be processed: ", imgs_to_process_paths)
        print("Reading polylines file...")
        raster = Dxf(config.polylines_path)
        for img_path in tqdm(imgs_to_process_paths):
            print("Creating image object..")
            image = AerialPicture(img_path, config.internals)
            print("Setting externals..")
            image.set_externals(config.dbfs_path)
            print("Creating mask..")
            mask = image.get_rooftop_mask(raster)
            
            if config.use_dsm:
                print("Getting depth mask..")
                depth_mask = image.get_depth_mask(config.dsm_path)
            else:
                depth_mask = None
            print("Preprocessing data..")
            cutted_mask, cutted_depth_mask, cutted_image = preprocess_mask_image(mask, image, depth_mask, config)
            img_id = image.img_basename

            save_img(cutted_mask, os.path.join(masks_path, img_id + ".png"))
            save_img(cutted_image, os.path.join(imgs_path, img_id + ".png"))
            if config.use_dsm:
               save_img(cutted_depth_mask, os.path.join(depths_path, img_id + ".png"))

def create_dataset(config):
    """
      The function uses as input the folder created during the preprocess function and creates
      the dataset folder ready to use.

      The function rescales the images,masks and depths, crops them and saves them in 
      working_dir/datasets/config.dataset_name, creating two folders: train and validation.
      Each split folder contains the folders images,masks and depths as described in the
      documentation of preprocess_data(config).
    """
    datasets_path = os.path.join(config.working_dir, "datasets")
    os.makedirs(datasets_path, exist_ok=True)
    dataset_path = os.path.join(datasets_path, config.dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    train_split_path = os.path.join(dataset_path, "train") 
    os.makedirs(train_split_path, exist_ok=True)
    val_split_path = os.path.join(dataset_path, "val") 
    os.makedirs(val_split_path, exist_ok=True)
    for split_path in (train_split_path, val_split_path):
       for data_type in ("images","masks","depths"):
          os.makedirs(os.path.join(split_path, data_type), exist_ok= True)
    
    preprocessed_data_path = os.path.join(config.working_dir, 
                                          "preprocessed_data", 
                                          config.preprocessed_dataset_name)
    images_names =  [os.path.basename(name).split(".")[0] for name in 
                     os.listdir(os.path.join(preprocessed_data_path, "imgs"))]
    val_names = config.val_names
    print(val_names)
    print(images_names)
    assert set(val_names).issubset(images_names)
    train_names = [name for name in images_names if name not in config.val_names]
    
    print("Validation set..")
    for name in tqdm(val_names):
        to_dataset_folder(name, config, "val")
    print("Train set..")
    for name in tqdm(train_names):
        to_dataset_folder(name, config, "train")


       
def load_images_masks(element_name, config):
  """Load the dataset images and the masks

  Parameters
  ----------
  element_name : str
      Name of the image to load without any extension 

  Returns
  -------
  img : np.ndarray
    Array of shape (1, H, W, 3), where (H,W) is the resolution
  mask : np.ndarray
    Array of shape (1, H, W, 1), where (H,W) is the resolution
  depth : np.ndarray
    Array of shape (1, H, W, 1), where (H,W) is the resolution

  """  
  
  preprocessed_path = os.path.join(config.working_dir, "preprocessed_data", config.preprocessed_dataset_name)
  img_path = os.path.join(preprocessed_path, "imgs", element_name + ".png")
  depth_path = os.path.join(preprocessed_path, "depths", element_name + ".png")
  mask_path = os.path.join(preprocessed_path, "masks", element_name + ".png")

  img = np.array(Image.open(img_path))
  mask = Image.open(mask_path)
  mask = mask.convert("L")
  mask = np.array(mask)

  if config.scale_factor != 1:
    mask = cv2.resize(mask, (mask.shape[1]//config.scale_factor, mask.shape[0]//config.scale_factor))
    mask = (mask > 0).astype(np.uint8)
    img = cv2.resize(img, (img.shape[1]//config.scale_factor, img.shape[0]//config.scale_factor))

  mask = mask[..., np.newaxis]
  mask = mask[np.newaxis, ...]
  img = img[np.newaxis, ...]
  depth = None
  
  if config.use_dsm:
    depth = Image.open(depth_path)
    depth = depth.convert("L")
    depth = np.array(depth)
    if config.scale_factor != 1:
      depth = cv2.resize(depth, (depth.shape[1]//config.scale_factor, depth.shape[0]//config.scale_factor)) 
    depth = depth[..., np.newaxis]
    depth = depth[np.newaxis, ...]

  return img, mask, depth


def crop_images_masks(images, masks, depths, crop_size, step):
  """Crop the given images and masks

  Parameters
  ----------
  images : np.ndarray
    Array of shape (N, H, W, 3), where N is the number of images and (H,W) the
    resolution
  masks : np.ndarray
    Array of shape (N, H, W, 1), where N is the number of images and (H,W) the
    resolution
  crop_size : int
      Height and width of the crops, i.e. C
  step : int
      Step used for generating the crops, i.e. the stride

  Returns
  -------
  cropped_images : np.ndarray
    Array of shape (M, C, C, 3), where M is the overall number of crops and 
    (C,C) the resolution of each crop
  cropped_masks : np.ndarray
    Array of shape (M, C, C, 1), where M is the overall number of crops and 
    (C,C) the resolution of each crop
  """
  h, w = images[0].shape[:-1]

  # Indices of the top-left crops corners
  crop_indices = get_crop_index(crop_size=crop_size, step=step, w=w, h=h)

  cropped_images = np.array([images[:, crop_indices[i][0]:crop_indices[i][0]+crop_size, crop_indices[i][1]:crop_indices[i][1]+crop_size, :] for i in range(len(crop_indices))])
  num_channels = cropped_images.shape[-1]
  cropped_images = np.reshape(cropped_images, (-1, crop_size, crop_size, num_channels))
  #infrared = cropped_images[..., -1]
  #cropped_images = cropped_images[..., :-1]  # TODO: removed infrared

  cropped_masks = np.array([masks[:, crop_indices[i][0]:crop_indices[i][0]+crop_size, crop_indices[i][1]:crop_indices[i][1]+crop_size, :] for i in range(len(crop_indices))])
  cropped_masks = np.reshape(cropped_masks, (-1, crop_size, crop_size, 1))

  #Removing samples without data
  flattened_data = cropped_masks.reshape(cropped_masks.shape[0], -1)
  crop_has_ones = np.any(flattened_data == 1, axis=1)
  valid_crop_indices = np.where(crop_has_ones)[0]
  cropped_images = cropped_images[valid_crop_indices]
  cropped_masks = cropped_masks[valid_crop_indices]
  
  if depths is not None:
    cropped_depths = np.array([depths[:, crop_indices[i][0]:crop_indices[i][0]+crop_size, crop_indices[i][1]:crop_indices[i][1]+crop_size, :] for i in range(len(crop_indices))])
    cropped_depths = np.reshape(cropped_depths, (-1, crop_size, crop_size, 1))
    cropped_depths = cropped_depths[valid_crop_indices]
  else:
    cropped_depths = None

  return cropped_images, cropped_masks, cropped_depths

def names_in_dataset(config):
   """
    Returns the list of the images file names without extension present in the specified dataset.
   """
   train_images_path = os.path.join(config.working_dir, "datasets", config.dataset_name, "train", "images")
   train_crops_names = os.listdir(train_images_path)
   val_images_path = os.path.join(config.working_dir, "datasets", config.dataset_name, "val", "images")
   val_crops_names = os.listdir(val_images_path)
   crops_names_unique = set(['_'.join(name.split("_")[:-1]) for name in train_crops_names + val_crops_names])
   return crops_names_unique

def to_dataset_folder(image_name, config, split):
  """
  Load a full image with relative ground truth and/or depth mask, crops them, and stores them.
  """
  if image_name in names_in_dataset(config):
    print(f"{image_name} already present in {config.dataset_name}, skipping")
    return
  print(f"{image_name}: Loading image and masks...")
  img, mask, depth = load_images_masks(image_name, config)
  print("Cropping..")
  cropped_image, cropped_mask, cropped_depth = crop_images_masks(img, mask, depth, config.crop_size, config.step)
  print("Saving..")
  split_path = os.path.join(config.working_dir, "datasets", config.dataset_name, split)
  for img_type, img_set in {"images": cropped_image, "masks": cropped_mask, "depths": cropped_depth}.items():
    if img_set is None:
       continue
    for i, image_array in enumerate(img_set):
        if img_type == "masks" or img_type == "depths":
          image_array = np.squeeze(image_array)
        image = Image.fromarray(image_array)
        os.makedirs(os.path.join(split_path, img_type), exist_ok = True)
        filename = f"{image_name}_{i}.png"
        image.save(os.path.join(split_path, img_type, filename))

def save_img(img, path):
    image = Image.fromarray(img)
    image.save(path)