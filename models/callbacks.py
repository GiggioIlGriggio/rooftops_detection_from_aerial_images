import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf

class DisplayCallback(tf.keras.callbacks.Callback):
    """Callback class for displaying the evolution of the predictions on a
    validation crop image during training.

    Parameters
        ----------
        val_crop_image : np.ndarray
            Validation crop image, shape (C, C, 3), where C is the crop size
        val_crop_mask : np.ndarray
            Validation crop mask, shape (C, C, 1), where C is the crop size
        epoch_interval : int, optional
            Interval of epochs on which displaying the predictions, by default 
            None (i.e. display every epoch)
    """
    def __init__(self, val_crop_image, val_crop_mask, epoch_interval=None):
        self.epoch_interval = epoch_interval
        self.val_crop_image = np.expand_dims(val_crop_image, axis=0)
        self.val_crop_mask = val_crop_mask

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            pred_crop_mask = self.model.predict(self.val_crop_image)[0]
            #pred_masks = tf.math.argmax(pred_masks, axis=-1)
            #pred_masks = pred_masks[..., tf.newaxis]

            # Randomly select an image from the test batch
            #random_index = random.randint(0, BATCH_SIZE - 1)
            #random_image = test_images[random_index]
            #random_pred_mask = pred_masks[random_index]
            #random_true_mask = test_masks[random_index]

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            ax[0].imshow(self.val_crop_image[0, :, :, :3])
            ax[0].set_title(f"Image: {epoch:03d}")

            ax[1].imshow(self.val_crop_mask)
            ax[1].set_title(f"Ground Truth Mask: {epoch:03d}")

            ax[2].imshow(pred_crop_mask)
            ax[2].set_title(
                f"Predicted Mask: {epoch:03d}",
            )

            plt.show()
            plt.close()