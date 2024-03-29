import os

ROOFTOP_IDS = [
    "31110000",
    "31110002",
    "3111000A",
    "3111000B",
    "3111000C",
    "3111000F",
    "3111000H",
    "3111000I",
    "3111000M",
    "3111000O",
    "3111000P",
    "3111000R",
    "3111000S",
    "311100AE",
    "311100CA",
    "311100CF",
    "311100DF",
    "311100FF",
    "311100ME",
    "311100MI",
    "311100MO",
    "311100PL",
    "311100PO",
    "311100SF",
    "311100SP",
    "31110100",
    "31110100"
]
ROOFTOP_IDS_REMOVE = [
    "31670000"
]

INTERNALS_DICT = {
    "TO1": {
        "focal_px" : 100.500 / 0.0046,
        "width" :  23010,
        "height" : 14790,
        "x_offset" : 0.,
        "y_offset" : 0.,
        "psize" : 0.0046
    }
}

class Config:
    def __init__(self,
                 #GENERAL
                 working_dir = "",      #Path to all the outputs of the model
                 checkpoints_dir = "",           #Name of the configuration
                 crop_size = 0,             #Size of the crops
                 model_name = "",            #"unet", "attunet", "unet_collection"
                 backbone = "",    #Backbone to use when "unet_collection" is selected
                 checkpoint = None,     #Path to model weights to upload
                 use_dsm = False,
                 seed = 33,        
                 #DATA PREPROCESSING
                 preprocessed_dataset_name = "",    #Name for the files after preprocessing
                 images_paths = [],     #List of all the paths of the images to use
                 internals = {},             #Dictionary of the internal parameters of the camera
                 dsm_path = "",         #Path to the folder containing all the LIDAR data
                 dbfs_path =  "",      #Path to the folder of dbf files containing external paramters
                 polylines_path = "",   #Path to the dxf file containing rooftop polygons
                 buffer_path = "",      #Path to the dxf file containing buffer polygon
                 depth_interpolation = True,    #Whether to interpolate the sparse matrix given by the LIDAR data
                 #DATASET CREATION
                 dataset_name = "",
                 val_names =  [],   #List of validation images names without extension
                 scale_factor = 1,          #Scale factor to rescale the images
                 step = 0,                  #Step size
                 #TRAINING
                 batch_size = 2,            
                 num_epochs = 30,            
                 pretrained = False,    #Use a pretrained backbone
                 freeze_backbone = True,#Freeze backbone during training
                 loss = "",             #String containing the losses to use separeted by a "_". They can be "tversky", "iou" or "binary". Ex "binary_iou"
                 #EVALUATION
                 ):
        
        self.checkpoints_dir = checkpoints_dir
        self.crop_size = crop_size
        self.step= step
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.model_name = model_name
        self.dbfs_path= dbfs_path
        self.dsm_path = dsm_path
        self.buffer_path = buffer_path
        self.depth_interpolation = depth_interpolation
        self.preprocessed_dataset_name = preprocessed_dataset_name
        self.working_dir = working_dir
        self.images_paths = images_paths
        self.loss = "_".join([l for l in ["tversky","iou","binary"] if l in loss])
        self.val_names = val_names
        self.polylines_path = polylines_path
        self.internals = internals
        self.backbone = backbone
        self.seed = seed
        self.freeze_backbone = freeze_backbone
        self.checkpoint = checkpoint
        self.num_epochs = num_epochs
        self.dataset_name = dataset_name
        self.use_dsm = use_dsm

        self.num_channels = 4 if self.use_dsm else 3
        self.pretrained = "imagenet" if pretrained and self.num_channels == 3 else None
