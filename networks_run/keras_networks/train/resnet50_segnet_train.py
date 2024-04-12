import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_segmentation.models import segnet

from training_base import TrainingBase

from tensorflow.keras.models import Model
from tensorflow.keras import layers

class PredefinedModel(TrainingBase):
    def __init__(
        self,
        train_txt_file_with_inputs,
        val_txt_file_with_inputs,
        dataset_root,
        model_output_path,
        learning_rate,
        number_of_epochs,
        width,
        height,
        batch_size,
        encoder_level,
    ) -> object:
        super().__init__(
            train_txt_file_with_inputs,
            val_txt_file_with_inputs,
            dataset_root,
            model_output_path,
            learning_rate,
            number_of_epochs,
            width,
            height,
            batch_size,
        )
        self.encoder_level = encoder_level

    def extend_model(self, original_model):
        # Use of downloaded model from https://github.com/divamgupta/image-segmentation-keras
        # Add a per-pixel classification layer
        cut_layer_idx = -3  # remove last convolutional layer and 2d upsampling layer
        cut_layer_name = original_model.layers[cut_layer_idx].name
        # Cut model
        model = Model(
            original_model.input,
            outputs=original_model.get_layer(cut_layer_name).output,
        )
        if self.encoder_level>2:
            for i in range(self.encoder_level-2):
                if i == 0:
                    model = layers.UpSampling2D(2, interpolation="bilinear")(
                        model.output
                    )
                else: 
                    model = layers.UpSampling2D(2, interpolation="bilinear")(
                        model
                    )
                model = layers.Conv2D(
                    4, kernel_size=(1, 1), activation="softmax", padding="same"
                )(model)
            model = Model(original_model.input,
                outputs=model)
        print(model.summary())
        return model

    def model_load(self):
        # Max num of encoder is equal to 4
        Segnet = segnet.resnet50_segnet(4, self.height, self.width, self.encoder_level)
        return self.extend_model(Segnet)


if __name__ == "__main__":
    # Initiate model with parameters
    model = PredefinedModel(
        train_txt_file_with_inputs=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/train_all_files.txt",
        val_txt_file_with_inputs = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/val_all_files.txt",
        dataset_root=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling",
        model_output_path=r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/model",
        learning_rate=0.0001,
        number_of_epochs=1,
        width=512,
        height=512,
        batch_size=2,
        encoder_level=5,
    )
    model.train_model()
