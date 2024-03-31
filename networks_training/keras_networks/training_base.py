import pathlib

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import Callback
import pandas as pd


class CustomCallbacks(Callback):
    """Class inheriting from A paCallback to create new custom callback for training

    """

    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        """This callback method will print actual learning rate

        Args:
            epoch (int): Number of actual epoch
            logs (object, optional): Previous logs. Defaults to None.
        """
        print('Current learning rate is:{:f}'.format(np.array(self.model.optimizer.learning_rate).item()))


class TrainingBase:
    """Base class for training of different neural networks
    """
    DEF_BACKBONE = ''

    def __init__(self,
                 txt_file_with_inputs,
                 dataset_root,
                 model_output_path,
                 model_filename,
                 val_coeficient,
                 learning_rate,
                 number_of_epochs,
                 image_size,
                 batch_size) -> object:
        self.model_output_path = model_output_path
        self.model_filename = model_filename
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.val_coeficient = val_coeficient
        self.image_size = image_size
        self.batch_size = batch_size
        list_of_files = open(txt_file_with_inputs).readlines()
        self.list_of_labels = [f"{dataset_root}/train{row.split(',')[1].strip()}" for row in list_of_files]
        self.list_of_iamges = [f"{dataset_root}/train{row.split(',')[0].strip()}" for row in list_of_files]

    def read_image(self, image_path, mask=False):
        """Read image and put it into tensor. If there is no mask provided, then preprocessing is done according to shape of resnet50

        Args:
            image_path (str): Path to image, which should be loaded
            mask (bool, optional): Split for mask images with labels and images for prediction. Defaults to False.

        Returns:
            tf.image: opened and read image
        """
        image = tf.io.read_file(image_path)
        if mask:
            image = tf.image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = tf.image.resize(images=image, size=[self.image_size, self.image_size])
            # image = tf.keras.utils.to_categorical(y = image, num_classes=5)[:,:,1:]
        else:
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = tf.image.resize(images=image, size=[self.image_size, self.image_size])
            image = tf.keras.applications.resnet50.preprocess_input(image)
        return image

    def load_data(self, image_list, mask_list):
        """Load both images. One for prediction and second image mask with classes

        Args:
            image_list (list): List of images for predictions paths
            mask_list (list): List of images class masks as annotations
            image_size (int): Size of the image that should be used

        Returns:
            tuple: tuple of list with loaded and read images
        """
        image = self.read_image(image_list)
        mask = self.read_image(mask_list, mask=True)
        return image, mask

    def data_generator(self, image_list, mask_list):
        """Read data with masks and images and stacked them into one tensor

        Args:
            image_list (list): List of image paths
            mask_list (list): List of label paths

        Returns:
            _type_: _description_
        """
        dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

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

    def model_load(self):
        raise NotImplementedError

    def train_model(self):
        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
        # Create output folder if doesnt exists
        if not pathlib.Path(self.model_output_path).exists():
            pathlib.Path(self.model_output_path).mkdir(parents=True)

        model_path = f"{self.model_output_path}/checkpoint.model.keras"
        train_images, val_images, train_masks, val_masks = train_test_split(
            self.list_of_iamges, self.list_of_labels, test_size=self.val_coeficient, random_state=42)
        # Put dataset into tensor
        train_dataset = self.data_generator(train_images, train_masks)
        val_dataset = self.data_generator(val_images, val_masks)

        model = self.model_load()

        # Define callback for saving model checkopoints
        checkpoint = ModelCheckpoint(model_path,
                                     monitor="val_accuracy",
                                     mode="min",
                                     save_best_only=True,
                                     verbose=1)

        # Define callback for reducing learning rate if no change in loss over specified period
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.2,
            patience=5,
            verbose=0,
            mode="auto",
            min_delta=1e-13,
            min_lr=1e-13)

        # Define callback for early stopping if no futher improvement reached
        earlystop = EarlyStopping(monitor='val_accuracy',  # value being monitored for improvement
                                  min_delta=1e-13,  # Abs value and is the min change required before we stop
                                  patience=20,  # Number of epochs we wait before stopping
                                  verbose=1,
                                  restore_best_weights=True)  # keeps the best weigths once stopped
        # Add tensorboard
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f"{self.model_output_path}/logs")

        # we put our call backs into a callback list
        callbacks = [earlystop, checkpoint, lr_scheduler, CustomCallbacks(), tensorboard_callback]

        # Compile model with define loss
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=["accuracy"],
        )

        # Start training
        history = model.fit(train_dataset,
                            validation_data=val_dataset,
                            callbacks=callbacks,
                            epochs=self.number_of_epochs,
                            batch_size=self.batch_size)
        pd.DataFrame(history.history).to_csv(f"{self.model_output_path}/logs/training_history.csv")
