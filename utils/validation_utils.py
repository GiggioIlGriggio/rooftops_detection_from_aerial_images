import numpy as np
import tensorflow as tf
import cv2
from tqdm.auto import tqdm
from PIL import Image
import os
from sklearn import metrics
import glob

from utils.utils import get_crop_index

Image.MAX_IMAGE_PIXELS = None

########################### PREDICT ON IMG
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_square(size, sigma):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D

def crop_map(img, index, crop_size):
    x0 = tf.constant(np.array(0, dtype=np.int32).reshape((1,)))
    b = tf.concat([index, x0], axis=0)
    n_channels = img.shape[2]
    crop = tf.slice(img, begin=b, size=tf.constant([crop_size, crop_size, n_channels]))
    crop = tf.divide(tf.cast(crop, tf.float32), 255.)
    #crop = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1)(crop)
    return crop, index

def predict_on_img(model, img_path, depth_mask_path, batch_size, crop_size, step, scale_factor, 
                   use_gaussian=False, unet3plus = False):
    """Compute the binary predictions on the full input image, returning a full 
    predicted mask.

    Basically, the image is splitted into small crops and the model is applied on
    each crop, returning the crop binary predictions. Then, all these crops masks 
    are merged together into a single full mask.

    The merging of the crops binary predictions is important, since the crops can 
    be overlapping. In other words, a pixel can receive more binary predictions.
    The final binary prediction on a pixel is simply computed as the average of 
    such several predictions.

    Optionally, the binary predictions computed on a crop can be rescaled by means
    of a gaussian filter, such that values computed far from the crop center are
    weakened (i.e. smaller values). The idea is that predictions near the crop
    center are reliable, and so they are not smoothened; but the far we go from
    the center, the less reliable are the values. However, a problem of this
    approach is that the predictions lose the semantic of beeing probabilities
    in the range [0,1]. For solving this, we could try to do a weighted average
    instead of a normal average. TODO

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained model
    img : np.ndarray [H, W, 3]
        Full input image
    batch_size : int
        Number of crops processed in paralell.
    crop_size : int
        Height and width of the crops, i.e. C
    step : int
        Step used for generating the crops, i.e. the stride
    use_gaussian : bool, optional
        Whether to use the gaussian rescaling or not, by default False

    Returns
    -------
    preds : np.ndarray [H, W, 1]
        Full predicted binary mask
    """

    img = np.array(Image.open(img_path))

    img = cv2.resize(img, (img.shape[1]//scale_factor, img.shape[0]//scale_factor))
    if depth_mask_path is not None:
        depth_mask = np.array(Image.open(depth_mask_path))
        depth_mask = cv2.resize(depth_mask, (depth_mask.shape[1]//scale_factor, depth_mask.shape[0]//scale_factor))
        depth_mask = np.expand_dims(depth_mask, axis=-1)
        img = np.concatenate((img,depth_mask), axis = -1)

    # gaussian weight
    # Gaussian filter
    sigma = crop_size/2
    gaussian_weight = gaussian_square(crop_size, sigma)
    gaussian_weight = np.repeat(gaussian_weight[np.newaxis, :], batch_size, axis=0)
    gaussian_weight = np.repeat(gaussian_weight[:, :, :, np.newaxis], 1, axis=-1)
    
    
    h, w = img.shape[:2]
    # Array [H, W, 1] in which we accumulate the per-pixel predictions
    preds = np.zeros((h, w, 1), dtype='float32')  # combined predictions
    # Array [H, W] in which we accumulate the per-pixel occurances, i.e. the 
    # number of predictions done for each pixel
    occs = np.zeros((h, w), dtype='uint8')  # pixel-wise prediction count

    # Indices of the top-left crops corners
    crop_indices = get_crop_index(crop_size=crop_size, step=step, w=w, h=h)

    # Generator which yields crops
    # TODO: make this function more efficient by replacing this generator with
    # the same approach used in `dataset_handler.py`
    gen = tf.data.Dataset.from_tensor_slices(np.array(crop_indices, dtype=np.int32)). \
        map(lambda x: crop_map(img, x, crop_size)). \
        batch(batch_size). \
        prefetch(buffer_size=5)

    # Iterate on each crop
    print(f"Running inference on {os.path.basename(img_path)}...")
    for crops, index in tqdm(gen):
        # Binary predictions on that crop
        crop_preds = model.predict(crops, verbose = 0)
        if unet3plus:
          crop_preds = crop_preds[-1]
        if use_gaussian: # Rescale using the gaussian filter
            crop_preds *= gaussian_weight[:crop_preds.shape[0]]
        # Update `preds` and `occs`
        for i in range(crop_preds.shape[0]):
            y_cur, x_cur = index[i].numpy()
            preds[y_cur:y_cur + crop_size, x_cur:x_cur + crop_size] += crop_preds[i]
            occs[y_cur:y_cur + crop_size, x_cur:x_cur + crop_size] += 1  # update prediction count

    # Let's compute the average
    occs = occs[..., np.newaxis]
    preds /= occs

    del occs

    return preds

def compute_metrics_all_val(model, config, save_inference = False):
    val_img_paths = [os.path.join(config.working_dir,"preprocessed_data",config.preprocessed_dataset_name,"imgs", name + '.png') for name in config.val_names]
    val_depth_paths = [os.path.join(config.working_dir,"preprocessed_data",config.preprocessed_dataset_name,"depths", name + '.png') for name in config.val_names]
    val_mask_paths = [os.path.join(config.working_dir,"preprocessed_data",config.preprocessed_dataset_name,"masks", name + '.png') for name in config.val_names]

    accuracies = []
    accuracies_fix = []
    accuracies_best = []
    accuracies_best_fix = []
    aurocs = []
    weights = []
    weights_effective = []

    if config.use_dsm is False:
        val_depth_paths = [None] * len(val_img_paths)

    for img_path, depth_path, mask_path in zip(val_img_paths,val_depth_paths,val_mask_paths):
        pred_rescaled = predict_on_img(model, img_path, depth_path, config.batch_size, config.crop_size, config.step, config.scale_factor)
        ground_truth = np.array(Image.open(mask_path))
        out_of_ROI = get_number_of_black_pixels(img_path)
        weight = ground_truth.shape[0] * ground_truth.shape[1]
        weight_effective = weight - out_of_ROI
        weights_effective.append(weight_effective)
        pred = cv2.resize(pred_rescaled, (ground_truth.shape[1], ground_truth.shape[0]))
        
        if save_inference:
            print("Saving predictions as png...")
            pred_save = pred * 255
            pred_binary = (pred > 0.5) * 255
            cv2.imwrite(os.path.join(config.working_dir,"checkpoints",config.checkpoints_dir, "pred_" + os.path.basename(img_path)), pred_save)
            cv2.imwrite(os.path.join(config.working_dir,"checkpoints",config.checkpoints_dir, "pred_binary_" + os.path.basename(img_path)), pred_binary)

        weights.append(weight)
        
        print("Computing auroc...")
        metrics_dict = compute_pixelwise_retrieval_metrics(np.expand_dims(pred, axis = 0), np.expand_dims(ground_truth, axis = 0))
        aurocs.append(metrics_dict["auroc"])
        
        print("Computing binary accuracy with optimal threshold...")
        optimal_th = metrics_dict["optimal_threshold"]
        acc_best = tf.keras.metrics.BinaryAccuracy(threshold=optimal_th)
        acc_best.update_state(ground_truth, pred)
        accuracies_best.append(acc_best.result().numpy())
        acc_best_fix = fix_acc(acc_best.result().numpy(), out_of_ROI, weight)
        accuracies_best_fix.append(acc_best_fix)

        print("Computing binary accuracy...")
        acc = tf.keras.metrics.BinaryAccuracy()
        acc.update_state(ground_truth, pred)
        accuracies.append(acc.result().numpy())
        acc_fix = fix_acc(acc.result().numpy(), out_of_ROI, weight)
        accuracies_fix.append(acc_fix)

    total_acc = 0 
    for acc, weight in zip(accuracies, weights):
        total_acc += weight * acc
    total_acc /= sum(weights)

    total_acc_fix = 0 
    for acc, weight in zip(accuracies_fix, weights_effective):
        total_acc_fix += weight * acc
    total_acc_fix /= sum(weights_effective)

    ret_dict = {"names": [os.path.basename(path) for path in val_img_paths],
            "accuracies": accuracies,
            "accuracies_fix": accuracies_fix,
            "accuracies_best": accuracies_best,
            "accuracies_best_fix": accuracies_best_fix, 

            "aurocs": aurocs, 
            "weights": weights,
            #"weights_effective": weights_effective,
            "total_acc": total_acc,
            "total_acc_fix": total_acc_fix
            }
    
    return ret_dict
               
def get_number_of_black_pixels(img_path):
    img = np.array(Image.open(img_path))
    black_pixels = np.sum(np.all(img == [0, 0, 0], axis=-1))
    return black_pixels

def fix_acc(acc, ooR, tot):
    y = acc * tot
    num = y  - ooR
    den = y - ooR + tot*(1-acc)
    return num/den

def compute_pixelwise_retrieval_metrics(predicted_segmentations, ground_truth_masks):
    """Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Function taken from the patchore anomaly detection model
    https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/metrics.py

    Parameters
    ----------
    predicted_segmentations : np.ndarray [NxHxW]
        Contains generated segmentation masks.
    ground_truth_masks : np.ndarray [NxHxW]
        Contains predefined ground truth segmentation masks

    Returns
    -------
    dict
        {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }
    """

    if isinstance(predicted_segmentations, list):
        predicted_segmentations = np.stack(predicted_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_predicted_segmentations = predicted_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_predicted_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_predicted_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_predicted_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_predicted_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }