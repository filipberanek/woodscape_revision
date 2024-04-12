import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_segmentation.models import pspnet

from training_base import TrainingBase


from tensorflow.keras.models import Model
from tensorflow.keras import layers


class PredefinedModel(TrainingBase):

    def extend_model(self, original_model, num_classes):
        # Use of downloaded model from https://github.com/divamgupta/image-segmentation-keras
        # Add a per-pixel classification layer
        cut_layer_idx = -3  # remove last convolutional layer and 2d upsampling layer
        cut_layer_name = original_model.layers[cut_layer_idx].name
        # Cut model
        model_cutted = Model(
            original_model.input,
            outputs=original_model.get_layer(cut_layer_name).output,
        )
        model_cutted = layers.UpSampling2D(2, interpolation="bilinear")(
            model_cutted.output
        )
        model_cutted = layers.Conv2D(
            num_classes, kernel_size=(1, 1), activation="softmax", padding="same"
        )(model_cutted)
        model_cutted = layers.UpSampling2D(2, interpolation="bilinear")(model_cutted)
        model_output = layers.Conv2D(
            num_classes, kernel_size=(1, 1), activation="softmax", padding="same"
        )(model_cutted)
        model = Model(original_model.input, model_output)
        print(model.summary())
        return model

    def model_load(self):
        Pspnet = pspnet.resnet50_pspnet(4, self.height, self.width)
        return self.extend_model(Pspnet, 4)


if __name__ == "__main__":
    # Initiate model with parameters
    model = PredefinedModel(
        train_txt_file_with_inputs=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/train_all_files.txt",
        val_txt_file_with_inputs = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/val_all_files.txt",
        dataset_root=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling",
        model_output_path=r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/model",
        val_coeficient=0.1,
        learning_rate=0.0001,
        number_of_epochs=1,
        width=576,
        height=384,
        batch_size=2,
    )
    model.train_model()
