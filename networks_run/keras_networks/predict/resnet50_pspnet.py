import sys
import os
import pathlib

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_pred_base import PredictBase
from keras_segmentation.models import pspnet
from tensorflow.keras.models import Model
from tensorflow.keras import layers


class PredictResnetUnet(PredictBase):

    def __init__(
        self,
        images_folder_path,
        model_path,
        output_path,
        width=576,
        height=384,
    ) -> object:
        super().__init__(images_folder_path, model_path, output_path, width, height)

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
        model_cutted = layers.UpSampling2D(2, interpolation="bilinear")(model_cutted.output)
        model_cutted = layers.Conv2D(num_classes, kernel_size=(1, 1), activation="softmax", padding="same")(
            model_cutted
        )
        model_cutted = layers.UpSampling2D(2, interpolation="bilinear")(model_cutted)
        model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), activation="softmax", padding="same")(
            model_cutted
        )
        model = Model(original_model.input, model_output)
        print(model.summary())
        return model

    def load_model(self):
        Pspnet = pspnet.resnet50_pspnet(4, self.height, self.width)
        return self.extend_model(Pspnet, 4)


if __name__ == "__main__":
    images_folder_path = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/rgbImages"
    model_path_root = pathlib.Path(
        r"/home/fberanek/Desktop/learning/my_articles/outputs/keras_networks/resnet50_pspnet_train_correct_clear_strict_files"
    )
    model_path = model_path_root / "model" / "checkpoint.model.h5"
    output_path = model_path_root / "predictions"
    PredictResnetUnet(
        images_folder_path=images_folder_path,
        model_path=model_path,
        output_path=output_path,
    ).predict()
