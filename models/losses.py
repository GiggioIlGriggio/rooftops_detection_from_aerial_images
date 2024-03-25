from keras_unet_collection import losses
import tensorflow as tf

def get_loss(loss_name):
    bc = tf.keras.losses.BinaryCrossentropy()

    def loss(y_true,y_pred):
        loss_items = []
        if "tversky" in loss_name:
            loss_items.append(losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3))
        if "iou" in loss_name:
            loss_items.append(losses.iou_seg(y_true, y_pred))
        if "binary" in loss_name:
            loss_items.append(bc(y_true, y_pred))
        if not loss_items:
            print("No valid loss given")
            return
        return sum(loss_items)
    
    return loss
