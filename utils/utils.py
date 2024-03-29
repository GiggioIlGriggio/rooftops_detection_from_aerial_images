from utils.config import ROOFTOP_IDS, ROOFTOP_IDS_REMOVE

import ezdxf
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import os
import itertools
from dbfread import DBF
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

Image.MAX_IMAGE_PIXELS = None

class Camera:

    def __init__(self, focal_px, width, height, x_offset, y_offset, psize):
        self.focal = focal_px
        self.width = width
        self.height = height 
        self.psize = psize
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.u0 = width / 2. + x_offset
        self.v0 = height / 2. + y_offset
        self.K = np.array([[-self.focal, 0, self.u0],
                    [0, -self.focal, self.v0],
                    [0, 0, 1]])
    
    def set_external(self, cam_coords, cam_rotation):
        """Setup external parameters of the camera. Each photo has different externals.

        Parameters
        ----------
        cam_coords : tuple of shape (EASTING, NORTHING, H_ORTHO)
            
        cam_rotation : tuple of shape (OMEGA, PHI, KAPPA)
            Expressed in degrees
        """
        self.t = np.array(cam_coords).reshape(3,1)
        self.omega, self.phi, self.kappa = np.radians(cam_rotation)

        self.R_x = np.array([[1, 0, 0],
                        [0, np.cos(self.omega), np.sin(self.omega)],
                        [0, -np.sin(self.omega), np.cos(self.omega)]])
        
        self.R_y = np.array([[np.cos(self.phi), 0, -np.sin(self.phi)],
                        [0, 1, 0],
                        [np.sin(self.phi), 0, np.cos(self.phi)]])
        
        self.R_z = np.array([[np.cos(self.kappa), np.sin(self.kappa), 0],
                        [-np.sin(self.kappa), np.cos(self.kappa), 0],
                        [0, 0, 1]])
        
        R_tmp = np.matmul(self.R_z, self.R_y)
        self.R = np.matmul(R_tmp, self.R_x)

    def world_to_camera_frame(self, x, y, z):
        """Converts a point coordinate from world frame to the camera frame

        Parameters
        ----------
        x : float
            Refers to EASTING
        y : float
            Refers to NORTHING
        z : float
            Refers to H_ORTHO

        Returns
        -------
        np.ndarray (3,1)

        """
        P_world = np.array([x, y, z]).reshape(3,1)
        P_camera = np.matmul(self.R,(P_world - self.t))
        return P_camera

    def camera_frame_to_pixel(self, P_camera):
        """Converts a point coordinate from the camera frame to the world frame

        Parameters
        ----------
        P_camera : np.ndarray (3,1)

        Returns
        -------
        uv: np.ndarray (2,1)
            Pixel coordinates of the point in the camera frame.
            u: x pixel coordinates
            v: y pixel coordinates
            The center of the image is in the top left corner
        """
        P_pixel = np.matmul(self.K, P_camera)
        uv = (P_pixel / P_pixel[2][0])[:-1]
        uv[1] = self.height - uv[1] #check if this is working with offset
        return uv
    
    def camera_frame_to_world(self, x, y, z):
        """The opposite of camera_world_to_frame
        """
        R_inv = np.linalg.inv(self.R)
        P_camera = np.array([x, y, z]).reshape(3,1)
        P_world = np.matmul(R_inv,P_camera) + self.t
        return P_world
    
    def get_coordinates_bb(self, world_depth): #world depth is aproximate, depends on the orientation of the camera (is referred to the camrea depth)
        """Get the bounding box of a rectangle at a specific world depth which enters in the camera

        Parameters
        ----------
        world_depth : float
    
        Returns
        -------
        tuple, tuple
            Two corners of the rectangle. (EASTING, NORTHING, H_ORTHO)
        """
        camera_depth = world_depth - self.t[2][0]
        right = self.width / 2 - self.x_offset
        left = - (self.width-right)
        top = self.height / 2 - self.y_offset
        bottom = - (self.height-top)

        bb_P = np.array([right, left, top, bottom])
        bb_C = bb_P * camera_depth / self.focal
        return (bb_C[0],bb_C[2],camera_depth), (bb_C[1],bb_C[3],camera_depth)

        
    def polyline_to_pixel(self, polyline):
        """Given a polyline expressed in world coordinates, it transforms it into camera pixel coordinates.

        Parameters
        ----------
        polyline : np.ndarray (N,3)
            Where N correspond to the number of points defining the polyline.
            The coordinates are in the order (EASTING, NORTHING, H_ORTHO)

        Returns
        -------
        np.ndarray (N,2)
            Theb coordinates are in the order (u,v) where u correspond to the x pixel coordinates and
            v to the y pixel coordinates. The center of the image is considered in the top left corner.
        """
        points_number = polyline.shape[0]
        pixels = []
        for i in range(points_number):
            P_camera = self.world_to_camera_frame(*tuple(polyline[i]))
            P_pixel = self.camera_frame_to_pixel(P_camera).flatten()
            pixels.append(P_pixel)
        return np.array(pixels)
    
class Dxf():
    def __init__(self, dxf_path):
        self.dxf = ezdxf.readfile(dxf_path)
    
    def get_polylines(self, ids, query, corner1, corner2):
        """Get all the polylines that fall into a rectangle identified by two corners in the
        world reference frame.

        Parameters
        ----------
        ids : List of strings
            Refer to the id of the polyline. Different ids refers to different structures. 
            ex. ["31110000"] refers to public building in our data.
        query : List of strings
            Refer to the type of lines in which we are interested.
            ex. ["LWPOLYLINE","POLYLINE"]
        corner1 : tuple (EASTING, NORTHING, H_ORTHO)
        corner2 : tuple (EASTING, NORTHING, H_ORTHO)

        Returns
        -------
        List of np.ndarray of shape (N,3) where N is the number of corners in the considered polyline

        """
        #Corner1 and corner2 must be in WRS
        min_x = min(corner1[0],corner2[0])
        min_y = min(corner1[1],corner2[1])
        max_x = max(corner1[0],corner2[0])
        max_y = max(corner1[1],corner2[1])
        polylines = []
        msp = self.dxf.modelspace()
        for q in query:
            for e in msp.query(q):
                if not ids or e.dxf.layer in ids:
                    if q == "POLYLINE":
                        for point in e.points():
                            if point[0] < max_x and point[0] > min_x and point[1] < max_y and point[1] > min_y:
                                polylines.append(np.array(list(e.points())))
                                break
                    if q == "LWPOLYLINE":
                        for point in e.vertices_in_wcs():
                            if point[0] < max_x and point[0] > min_x and point[1] < max_y and point[1] > min_y:
                                polylines.append(np.array(list(e.vertices_in_wcs())))
                                break
        return polylines
    
    def get_buffer(self):
        """
        Get the polyline of the buffer. It is used in order to mask the regions in which ground truth data
        is not avaiable.
        The dxf must be a 1 polygon buffer.

        Returns
        -------
        np.ndarray (N,3) where N is the number of corners in the polyline
        """
        msp = self.dxf.modelspace()
        if len(msp) != 1:
            print("The dxf is not a buffer")
            return
        return np.array(list(msp[0].points())) 

class Dsm():
    """
    Class that describes the LIDAR data.
    """
    def __init__(self, tif_path):
        tfw_path = tif_path.replace(".tif", ".tfw")
        self.x_pixel_size,  _, _, self.y_pixel_size, self.easting, self.northing = self.read_tfw(tfw_path)
        self.tif_model = np.array(Image.open(tif_path))
        self.height, self.width = self.tif_model.shape

    def read_tfw(self, filename):
        with open(filename, 'r') as file:
            raws = file.readlines()
        return [float(raw.strip()) for raw in raws]
    
    def get_parallelepiped_vertices(self):
        """
        Get the two corners of the prallelepiped which contains all the LIDAR points.

        Returns
        -------
        np.ndarray (2,3)
            Where 2 is the number of the corners and 3 refers to the corners coordinates in world reference
            frame (EASTING, NORTHING, H_ORTHO)

        """
        max_height = np.max(self.tif_model)
        self.tif_model = np.where(self.tif_model == -32767., np.inf, self.tif_model)
        min_height = np.min(self.tif_model)
        self.tif_model = np.where(self.tif_model == np.inf, -32767., self.tif_model)
        min_x = self.easting
        min_y = self.northing
        max_x = min_x + self.x_pixel_size * self.width
        max_y = min_y + self.y_pixel_size * self.height

        x_set = {min_x, max_x}
        y_set = {min_y, max_y}
        z_set = {min_height, max_height}

        parallelepiped_vertices = np.array(list(itertools.product(x_set, y_set, z_set)))

        return parallelepiped_vertices

class AerialPicture(): # TODO implement function that looks for the internals paramters
    def __init__(self, img_path, cam_internals):
        self.img_basename = os.path.basename(img_path).split(".")[0]
        self.image = np.array(Image.open(img_path))
        self.cam = Camera(**cam_internals)
        self.depth_mask = None
        self.image_plane_polygon = Polygon([[0,0],
                                            [0,self.cam.width],
                                            [self.cam.height,self.cam.width],
                                            [self.cam.height,0]])

    def set_externals(self, dbfs_path):
        self.cam.set_external(*self.get_external_parameters(dbfs_path))

    def get_external_parameters(self, dbfs_folder_path):
        """Get the external parameters for the considered image

        Parameters
        ----------
        dbfs_folder_path : String
            Path that points to the path which contains all the dbf files. Each dbf file contains the external
            parameters of multiple images.

        Returns
        -------
        Tuple, Tuple
            cam_coords are the coordinates of the camera center in world reference frame 
                (EASTING, NORTHING, HORTHO)
            cam_rotations are the rotations of the camera reference frame wrt world reference frame
                (OMEGA, PHI, KAPPA)
        """
        dbf_file_names = [name for name in os.listdir(dbfs_folder_path) if ".dbf" in name]
        for dbf_name in dbf_file_names:
            dbf_path = os.path.join(dbfs_folder_path, dbf_name)
            table = DBF(dbf_path)
            for record in table:
                if record['ID_FOTO'] == self.img_basename:
                    print(f"External parameters for image {self.img_basename} found at {dbf_path}")
                    record_dict = dict(record)
                    cam_coords = (record_dict["EASTING"], record_dict["NORTHING"], record_dict["H_ORTHO"])
                    cam_rotations = (record_dict["OMEGA"], record_dict["PHI"], record_dict["KAPPA"])
                    return cam_coords, cam_rotations
        print("Image not found in the provided folder")
    
    def get_depth_mask(self, models_folder = None):
        if self.depth_mask:
            return self.depth_mask
        elif models_folder is None:
            print("self.epth_mask not build, and models_folder not provided to build it")
            return
        else:
            ("Building depth_mask...")
            self.depth_mask = self.build_depth_mask(models_folder)
            return self.depth_mask

    def build_depth_mask(self, models_folder):
        """
        Builds the mask of the iamge using the values from the LIDAR data.
        Projects each point identified by the LIDAR (x, y, z) to the image plane, thus creating
        a mask. The values of the mask are given by z.
        The resulting mask is a sparse matrix.

        Parameters
        ----------
        models_folder : String
            Path pointing to the folder containing all the tif and tfw files of the LIDAR data.

        Returns
        -------
        np.ndarray (W,H,1) of type np.float32
        """
        valid_models = self.get_valid_models(models_folder)
        if valid_models == []:
            print(f"No valid model found for the image {self.img_basename}")
        mask = np.zeros(self.image.shape[:2], dtype=np.float32)
        print(f"Found {len(valid_models)} valid LIDAR models, starting projections...")
        for model in valid_models:
            for x in range(model.width):   #TODO improve efficiency (remove for loops)
                for y in range(model.height):
                    if model.tif_model[y,x] < 0:
                        continue
                    easting_p = model.easting + x * model.x_pixel_size
                    northing_p = model.northing + y * model.y_pixel_size #(height - y) * y_pixel_size
                    ortho_h_p = model.tif_model[y,x]

                    P_camera = self.cam.world_to_camera_frame(easting_p, northing_p, ortho_h_p)
                    P_pixel = self.cam.camera_frame_to_pixel(P_camera).flatten()
                    
                    y_p = int(P_pixel[1])
                    x_p = int(P_pixel[0])

                    if y_p>=0 and y_p < self.image.shape[0] and x_p >=0 and x_p < self.image.shape[1]:
                        mask[y_p,x_p] = max(mask[y_p,x_p], ortho_h_p)
        return mask

    def get_valid_models(self, models_folder):
        """Get the LIDAR data which could contain a point in the considered image.

        Parameters
        ----------
        models_folder : String
            Path point to the LIDAR data folder

        Returns
        -------
        List of Dsm
        """
        valid_models = []
        models_paths = [os.path.join(models_folder, tif_path) for tif_path in os.listdir(models_folder) if ".tif" in tif_path]
        for tif_path in models_paths:
            model = Dsm(tif_path)
            parallelepiped_vertices = model.get_parallelepiped_vertices()
            parallelepiped_vertices_pixel = self.cam.polyline_to_pixel(parallelepiped_vertices)
            parallelepiped_polygon = Polygon(parallelepiped_vertices_pixel)
            if parallelepiped_polygon.intersects(self.image_plane_polygon):
                valid_models.append(model)
        return valid_models

    def get_rooftop_mask(self, raster):
        """Retrive the relative mask of the rooftops.

        Parameters
        ----------
        raster : Dxf

        Returns
        -------
        np.ndarray of type np.uint8
            Mask in which 1 is rooftop, 0 is not rooftop.
        """
        corner1, corner2 = self.cam.get_coordinates_bb(world_depth= 0)
        corner1, corner2 = self.cam.camera_frame_to_world(*corner1), self.cam.camera_frame_to_world(*corner2)
        polylines = raster.get_polylines(ROOFTOP_IDS, ["POLYLINE","LWPOLYLINE"], corner1, corner2)
        polylines = [self.cam.polyline_to_pixel(polyline) for polyline in polylines]
        polylines_to_remove = raster.get_polylines(ROOFTOP_IDS_REMOVE, ["POLYLINE","LWPOLYLINE"], corner1, corner2)
        polylines_to_remove = [self.cam.polyline_to_pixel(polyline) for polyline in polylines_to_remove]
        return get_mask(polylines, polylines_to_remove, self.cam.width, self.cam.height, show= False, objects_value = 1)

def get_mask(polylines, polylines_to_remove, img_width, img_height, show = False, objects_value = 1):
    """Creates the mask by filling the polylines of the rooftops.

    Parameters
    ----------
    polylines : List of np.ndarray (N,2)
        List of rooftop polylines expressed in pixel coordinates
    polylines_to_remove : List of np.ndarray (N,2)
        List of not rooftop polylines that are contained into the rooftops expressed in pixel coordinates
    img_width : Integer
        Width of the image
    img_height : Integer
        Height of the image
    show : bool, optional
        Plot the mask, by default False
    objects_value : int, optional
        Value given to the rooftop pixels in the mask, by default 1

    Returns
    -------
    np.ndarray (H,W,1) of dtype np.uint8

    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for polyline_pixel in polylines:
        cv2.fillPoly(mask, np.int32(np.expand_dims(polyline_pixel, axis=0)), objects_value)
    for polyline_pixel in polylines_to_remove:
        cv2.fillPoly(mask, np.int32(np.expand_dims(polyline_pixel, axis=0)), 0)
    if show:
        plt.imshow(mask,cmap='gray')
        plt.show()
        plt.close()
    return mask

def shadow_image_mask(aerialImage, mask, depth_mask, buffer_path):
    """
    Set to 0 all the pixels that fall outside of the buffer. The buffer represents the region of 
    interest in which ground truth data is avaible.

    Parameters
    ----------
    aerialImage : AerialImage

    mask : np.ndarray (H,W,1)

    depth_mask : np.ndarray (H,W,1)

    buffer_path : String
        Path pointing to the .dxf file of the buffer

    Returns
    -------
    Tuple : (image, mask, depth_mask)
        
    """
    dxf = Dxf(buffer_path)
    buffer_pixel = aerialImage.cam.polyline_to_pixel(dxf.get_buffer())
    image = aerialImage.image.copy()
    height, width = image.shape[:-1]
    mask_buffer = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_buffer, np.int32(np.expand_dims(buffer_pixel, axis=0)), 1)
    image[mask_buffer == 0] = [0, 0, 0]
    mask[mask_buffer == 0] = [0]
    if depth_mask is not None:
        depth_mask[mask_buffer == 0] = [0]
    return image, mask, depth_mask

def normalize_non_zero_elements(el):
    """
    Normalize all non zero elements in a np.ndarray
    """
    non_zero_ids = el.nonzero()
    max = np.max(el[non_zero_ids])
    min = np.min(el[non_zero_ids])
    el[non_zero_ids] = (el[non_zero_ids] - min)/(max - min)
    return el

def preprocess_mask_image(mask, aerialImage, depth_mask, config):
    """Preprocess of the images.
    Set to 0 data without ground truth.
    Cut image, mask, and depth mask to the smallest size which contains ground truth data.
    If the LIDAR data is used, it normalize the data and eventually performs interpolation
    on the sparse values.

    Parameters
    ----------
    mask : np.ndarray (H,W,1)
    aerialImage : AerialImage
    depth_mask : np.ndarray (H,W,1)
        It is a sparse matrix
    config : Config

    Returns
    -------
    Tuple
        mask_cut : np.ndarray (H,W,1) of type np.uint8
        depth_mask_cut : np.ndarray (H,W,1) of type np.uint8
        image_cut : np.ndarray (H,W,3) of type np.uint8
    """
    image, mask, depth_mask = shadow_image_mask(aerialImage, mask, depth_mask, config.buffer_path)
    
    height, width = mask.shape
    offset = 100
    top_index = min(np.where(mask.any(axis=1))[0][-1] + offset, height)
    bottom_index = max(np.where(mask.any(axis=1))[0][0] - offset, 0)
    left_index = max(np.where(mask.any(axis=0))[0][0] - offset, 0)
    right_index = min(np.where(mask.any(axis=0))[0][-1] + offset, width)

    mask_cut = mask[bottom_index:top_index, left_index:right_index]
    image_cut = image[bottom_index:top_index, left_index:right_index,:]
    
    if depth_mask is not None:
        depth_mask_cut = depth_mask[bottom_index:top_index, left_index:right_index]
        depth_mask_cut = normalize_non_zero_elements(depth_mask_cut)
        print(depth_mask_cut.shape)
        print(depth_mask_cut.dtype)
        print(np.count_nonzero(depth_mask_cut))
        if config.depth_interpolation:
            coords = np.column_stack(np.nonzero(depth_mask_cut))
            values = depth_mask_cut[coords[:, 0], coords[:, 1]]
            grid_x, grid_y = np.mgrid[0:depth_mask_cut.shape[0], 0:depth_mask_cut.shape[1]]
            depth_mask_cut = griddata(coords, values, (grid_x, grid_y), method='cubic')
        depth_mask_cut = (depth_mask_cut * 255).astype(np.uint8)
    else:
        depth_mask_cut = None

    return mask_cut, depth_mask_cut, image_cut


def get_crop_index(crop_size, step, w, h):
    """Return the crops indices.
    In particular, it returns the indices of the top-left corner of the crops.

    Parameters
    ----------
    crop_size : int
        Height and width of the crops, i.e. C
    step : int
        Step used for generating the crops, i.e. the stride
    w : int
        Width of the original image
    h : int
        Height of the original image

    Returns
    -------
    crop_indices : list of list
        List of couples (y_i, x_i), representing the coordinates of the top-left
        corner of each crop.
        So, the length of `crop_indices` is M, which is the total number of crops.
    """
    y_cur = 0  # upper left
    crop_indices = []

    while y_cur + crop_size < h:  # top-down

        x_cur = 0
        crop_indices += [[y_cur, x_cur]]

        while x_cur + crop_size < w:  # left to right
            if x_cur + step + crop_size < w:
                x_cur += step
            else:
                x_cur = w - crop_size
            crop_indices += [[y_cur, x_cur]]

        if y_cur + step + crop_size < h:
            y_cur += step
        else:  # y of last row
            y_cur = h - crop_size
            x_cur = 0
            crop_indices += [[y_cur, x_cur]]

        # last row
        while x_cur + crop_size < w:
            if x_cur + step + crop_size < w:
                x_cur += step
            else:
                x_cur = w - crop_size
            crop_indices += [[y_cur, x_cur]]
    return crop_indices