import tensorflow as tf
import os
import datetime
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def get_callbacks(experiment_name="", model_name="", is_fine_tune=False):
    if not experiment_name or not model_name:
        logging.error("You didn't specify the experimnet and model names for creating callbacks.")
        return None

    try: 
        os.makedirs(experiment_name, exist_ok=True)
        
        log_dir = os.path.join(experiment_name, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        checkpoint_dir = os.path.join(experiment_name, "checkpoints", model_name)
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    except OSError as e:
        logging.error("Failed to create directories: {e}")
        return None
    
    #Create Early Stopping, LR shedular and checkpoint callbacks:
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    lr_shedular = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir+'best_model.keras', monitor="val_loss", save_best_only=True)

    return early_stopping, lr_shedular, checkpoint
