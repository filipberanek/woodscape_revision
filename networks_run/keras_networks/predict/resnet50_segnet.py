import sys
import os
import pathlib

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_pred_base import PredictBase
from keras_segmentation.models import segnet
from tensorflow.keras.models import Model
from tensorflow.keras import layers


class PredictResnetUnet(PredictBase):

    def __init__(
        self,
        images_folder_path,
        model_path,
        output_path,
        width=512,
        height=512,
        encoder_level=3,
    ) -> object:
        super().__init__(images_folder_path, model_path, output_path, width, height)
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
        if self.encoder_level > 2:
            for i in range(self.encoder_level - 2):
                if i == 0:
                    model = layers.UpSampling2D(2, interpolation="bilinear")(model.output)
                else:
                    model = layers.UpSampling2D(2, interpolation="bilinear")(model)
                model = layers.Conv2D(4, kernel_size=(1, 1), activation="softmax", padding="same")(model)
            model = Model(original_model.input, outputs=model)
        print(model.summary())
        return model

    def load_model(self):
        Segnet = segnet.resnet50_segnet(4, self.height, self.width, self.encoder_level)
        return self.extend_model(Segnet)


if __name__ == "__main__":
    images_folder_path = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/rgbImages"
    model_path_root = pathlib.Path(
        r"/home/fberanek/Desktop/learning/my_articles/outputs/keras_networks/resnet50_segnet_train_all_files"
    )
    model_path = model_path_root / "model" / "checkpoint.model.h5"
    output_path = model_path_root / "predictions"
    PredictResnetUnet(
        images_folder_path=images_folder_path,
        model_path=model_path,
        output_path=output_path,
    ).predict()
