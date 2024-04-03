import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np

from keras_segmentation.models import unet


class Prediction:
    """Base class for training of different neural networks
    """

    def __init__(self,
                 images_folder_path,
                 model_path,
                 output_path,
                 image_size) -> object:
        self.images_folder_path = images_folder_path
        self.model_path = model_path
        self.output_path = pathlib.Path(output_path)
        self.image_size = image_size

    def extend_model(self, original_model, num_classes, image_size):
        # Use of downloaded model from https://github.com/divamgupta/image-segmentation-keras
        # Add a per-pixel classification layer
        cut_layer_idx = -3  # remove last convolutional layer and 2d upsampling layer
        cut_layer_name = original_model.layers[cut_layer_idx].name
        # Cut model
        model_cutted = Model(original_model.input, outputs=original_model.get_layer(cut_layer_name).output)
        model_cutted = layers.UpSampling2D(2, interpolation="bilinear")(model_cutted.output)
        model_output = layers.Conv2D(
            num_classes, kernel_size=(1, 1),
            activation="softmax", padding="same")(model_cutted)
        model = Model(original_model.input, model_output)
        print(model.summary())
        return model

    def load_model(self):
        Unet = unet.resnet50_unet(4, self.image_size, self.image_size, 3)
        return self.extend_model(Unet, 4, self.image_size)

    def train_model(self):
        lo_images = pathlib.Path(self.images_folder_path).glob("*.png")
        model = self.load_model()
        model.load_weights(self.model_path)
        # Proces predictions
        for image_path in tqdm(lo_images):
            # Load image and mask
            img = Image.open(image_path)
            img = img.resize((self.image_size, self.image_size))
            image_tensor = tf.constant(img)
            predictions = model.predict(np.expand_dims((image_tensor), axis=0))
            predictions = np.squeeze(predictions)
            prediction_mask = np.argmax(predictions, axis=-1)
            pred_image = Image.fromarray(prediction_mask)
            pred_image.save(self.output_path/image_path.name)


images_folder_path = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/rgbImages"
model_path = r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/model/checkpoint.model.h5"
output_path = r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/predictions"
Prediction(images_folder_path=images_folder_path,
           model_path=model_path,
           output_path=output_path,
           image_size=512).train_model()
