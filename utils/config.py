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

class Config:
    def __init__(self,
                 config_name,
                 crop_size,
                 step,
                 scale_factor,
                 internals,
                 batch_size,
                 model_name,
                 num_epochs,
                 backbone = "VGG16",
                 pretrained = False,
                 freeze_backbone = True,
                 checkpoint = None,
                 dsm_paths = [],
                 dbfs_paths = [],
                 polylines_path = "",
                 buffer_path = "",
                 depth_interpolation = True,
                 working_dir = "",
                 preprocessed_dataset_name = "",
                 images_paths = [],
                 loss = "",
                 val_names =  ['05_26_1892'],
                 seed = 33
                 ):
        
        self.config_name = config_name
        self.crop_size = crop_size
        self.step= step
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.model_name = model_name
        self.dbfs_paths= dbfs_paths
        self.dsm_paths = dsm_paths
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
        
        self.use_dsm = True if dsm_paths else False
        self.num_channels = 4 if self.use_dsm else 3
        self.depth_name = "nodepth" if not self.use_dsm else "depthi" if self.depth_interpolation else "depth"
        self.dataset_name = f"{crop_size}_{step}_{scale_factor}_{self.depth_name}"
        self.pretrained = "imagenet" if pretrained and self.num_channels == 3 else None
