# rooftops-building-detection

Segmentation of rooftop buildings from aerial images and LIDAR data.

## Environment

Create and activate an environment with inside `requirements.txt`.

## Functionalities

1. Data preprocessing
2. Dataset creation
3. Model training
4. Model evaluation

### 1. Data preprocessing

It is rapresented by the function `preprocess_data` located at `utils/dataset_handler.py` .

The function takes the input files, such as the images and the related camera parameters, the dxf file containing the rooftops polylines, the dxf file identifying the raster and the tif files relative to the LIDAR data.

The function calls all the projections functions to produce the output.

The preprocessing consists in:

* matching images, camera parameters, rooftop polylines and LIDAR data if avaible.
* cutting the images, masks and LIDAR data to the region where ground truth data is avaible. Since the depth mask given from the LIDAR data has less resolution, it can be interpolated.

The output is stored in `config.working_dir/preprocessed_data/config.preprocessed_dataset_name `and it consist of 3 folders:

* `imgs`: contains the RGB images
* `masks`: contains the rooftops ground truth masks
* `depths`: contains the masks of LIDAR data

### 2. Dataset creation

It is rapresented by the function `create_dataset` located at `utils/dataset_handler.py` .

Since cropping the images and storing all of them in the GPU RAM is too much memory consuming, the aim of this function is to save the crops in the drive, so that later can be load at runtime during training.

The function uses as input the folder created during the data preprocessing step and creates the dataset folder ready to use. The function rescales the images, masks and depths, crops them and saves them in config.working_dir/datasets/config.dataset_name, creating two folders: train and validation. Each split folder contains the folders `images`, `masks `and `depths `as described in the previous section.

### 3. Model training

It is rapresented by the function `run_training` located in `run.py` .

### 4. Model evaluation

It is rapresented by the function `compute_metrics_all_val` located in `utils/validation_utils.py` .

It performs an evaluation of the selected model on the validation data, retriving the binary accuracy of the model.

## Example usage

First intall requirements:

```python
pip install -r requirements.txt
```

Data preprocessing

```
python preprocess_data.py working_dir TEST_preprocess_dataset_name images_TEST TO1 --use_dsm 05.04_DDSM_1M dbfs DTP_TO_minuta_di_restituzione_01.dxf buffer1.dxf --depth_interpolation
```

`preprocess_data.py` arguments:

* `working_dir`: String of the path to an existing folder. In this folder will be stored all the results of the scripts.
* `preprocessed_dataset_name`: String of the name of the preprocessed data.
* `images_paths`: List of String containing the paths of the  images to preprocess.
* `internals` : String with the ID of the internal parameters to use. Implemented only "TO1".
* `--use_dsm` : Bool wheter to use LIDAR data or not.
* `dsm_path`: String of the path to the folder containing all dsms (LIDAR data) in .tif format.
* `dbfs_path`: String of the path to the folder of dbf files containing external paramters.
* `polylines_path`: String of the path to the .dxf file containing the rooftop polygons.
* `buffer_path`: String of the path to the .dxf file containing the buffer polygon.
* `--depth_interpolation`: Bool if wheter or not interpolate the depth images.

Dataset creation

```
python create_dataset.py working_dir TEST_dataset 256 128 TEST_preprocess_dataset_name 4 --use_dsm --val_names 02_65_1802
```

`create_dataset.py` arguments:

* `working_dir`
* `dataset_name`: String of name of the dataset.
* `crop_size`
* `step`
* `preprocessed_dataset_name`: String of the name of the preprocessed data
* `scale_factor`
* `--use_dsm`: Bool, whether to include LIDAR data or not in the final dataset
* `--val_names`: List of strings with the name of the images to be used as validation without extension.

Training

```
python run.py working_dir unet TEST_dataset 256 --use_dsm 2 2 binary TEST_checkpoint_dir
```

`run.py` arguments:

* `working_dir`
* `model_name`: String, it can be "unet" or "attunet"
* `dataset_name`
* `crop_size`
* `--use_dsm`
* `batch_size`
* `num_epochs`
* `loss`: String containing the losses to use separeted by a "_". They can be `"tversky"` for Tversky focal loss, `"iou"` for a generalized iou loss or `"binary"` for the BinaryCrossEntropy loss. Ex `"binary_iou"` will use a combination of the `"binary"` and `"iou"` loss.
* `checkpoints_dir`: String, where to store the model weights

Inference

```
python inference.py unet working_dir/checkpoints/TEST_checkpoint_dir/unet.02-0.7798.hdf5 working_dir/preprocessed_data/TEST_preprocess_dataset_name/imgs/02_65_1802.png working_dir/preprocessed_data/TEST_preprocess_dataset_name/depths/02_65_1802.png working_dir 256 4 --use_dsm
```

`inference.py` arguments:

* `model_name`:"unet" or "attunet"
* `checkpoint`: path to the pretrained weigths
* `img_path`: path to the image to inference on.
* `depth_mask_path`: path to the relative depth mask
* `output_path`: String of path, where to store the predictions
* `crop_size`
* `scale_factor`
* `--use_dsm`: Bool, if set to False will ignore `depth_mask_path`

## Understand the arguments

The core structure which manages the repo is the Config class, in which all the arguments are contained.

General arguments:

* `working_dir`: String of the path to an existing folder. In this folder will be stored all the results of the scripts.

  ```
  working_dir
  ├── checkpoints
  ├── datasets
  └── preprocessed_data
  ```
* `checkpoints_dir`: String of the name of the configuration. the checkpoints of the models will be saved in working_dir/checkpoints/checkpoints_dir
* `crop_size`: Integer of the crop size. The input size of the model will be (batch_size, crop_size, crop_size, n_channels)
* `model_name`: "unet" or "attunet"
* `checkpoint`: None if model should be initialized with random weights, or String of the path to the .hdf5 file containing pretrained weights.
* `use_dsm`: Bool, wheter to use LIDAR data or not. If this argument is set to False for preprocess of data or during dataset creation it will ignore the LIDAR data provided. If it is used during training or evaluation, the scripts will generate RGB inputs for the model without the LIDAR data as 4-th channel.
* `seed`

Data preprocessing arguments:

* `preprocessed_dataset_name`: String of the name of the preprocessed data. The preprocessed data will be stored in `working_dir/preprocessed_data/preprocessed_dataset_name`
* `images_paths`: List of String containing the paths of the  images to preprocess. Supported image extension is `.tif`. All images must share internal parameters of the camera.
* `internals`: Dict of the internal parameters of the camera with the following format.

  ```
  {
      "focal_px" : 100.500 / 0.0046,    	focal length in pixels
      "width" :  23010,	 		focal width in pixels
      "height" : 14790,			focal height in pixels
      "x_offset" : 0.,			x offset of the camera center
      "y_offset" : 0.,			y offset of the camera center
      "psize" : 0.0046			pixel size in mm
      }
  ```
* `dsm_path`: String of the path to the folder containing all dsms (LIDAR data) in .tif format. The corresponding .tfw should be provided in the same folder with the same name of the .tif file. In the .tif file the pixels containing no data have the value of -32767. The format for the .tfw file should be the following

  ```
  1		pixel size along x-axis
  0		//
  0		//
  -1		pixel size along y-axis
  6671630.5	easting coordinate of the top left pixel
  5006799.5	northing coordinate of the top left pixel
  ```
* `dbfs_path`: String of the path to the folder of dbf files containing external paramters. Each dbf table can contain multiple records. Each record contains the external parameters of a single image identified by the "ID_FOTO" which corresponds to the name of the image without extension. The format for the dbf file should be the following:

  ```
  record["ID_FOTO"]	name of the image without extension
  record["EASTING"]	camera coordinates in world reference frame
  record["NORTHING"]
  record["H_ORTHO"]
  record["OMEGA"]		camera rotations
  record["PHI"]
  record["KAPPA"]
  ```
* `polylines_path`: String of the path to the .dxf file containing the rooftop polygons. The lines which are used are "POLYLINES" and "LWPOLYLINES". The polygons ids relative to the rooftops are stored in the global varible `ROOFTOP_IDS `in the file `utils/config.py`. The global variable `ROOFTOP_IDS_REMOVE` contains the ids of the cavaedium polygons.
* `buffer_path`: String of the path to the .dxf file containing the buffer polygon. This is used to set to 0 all the pixels in images,masks and dpeths that fall out of the buffer, thus they don't have ground truth data.
* `depth_interpolation`: Bool if wheter or not interpolate the depth image. In fact LIDAR data has less density with respect to the aerial images and, by only projecting, it will create a sparse matrix.

Dataset creation:

* `val_names`: List of Strings containing the images names without extension that should be used as validation
* `scale_factor`: Integer for the rescaling factor of the data. For example `scale_factor = 4` means that input full images, masks and eventually depths are rescaled down by a factor of 4 (ex. 32x32 -> 8x8), and then are cropped to the crop_size.
* `step`: Integer of the step size.
* `dataset_name`: String of name of the dataset.

Training:

* `batch_size`
* `num_epochs`
* `loss`: String containing the losses to use separeted by a "_". They can be `"tversky"` for Tversky focal loss, `"iou"` for a generalized iou loss or `"binary"` for the BinaryCrossEntropy loss. Ex `"binary_iou"` will use a combination of the `"binary"` and `"iou"` loss.
