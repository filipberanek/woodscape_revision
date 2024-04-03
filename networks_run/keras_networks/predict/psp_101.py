import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_pred_base import PredictBase
from keras_segmentation.models import pspnet

from tensorflow.keras.models import Model


class PredictResnetUnet(PredictBase):

    def extend_model(self, original_model, num_classes, image_size):
        # Use of downloaded model from https://github.com/divamgupta/image-segmentation-keras
        # Add a per-pixel classification layer
        cut_layer_idx = -3  # remove last convolutional layer and 2d upsampling layer
        cut_layer_name = original_model.layers[cut_layer_idx].name
        # Cut model
        model = Model(
            original_model.input,
            outputs=original_model.get_layer(cut_layer_name).output,
        )
        print(model.summary())
        return model

    def load_model(self):
        Pspnet = pspnet.pspnet_101(4, self.image_size, self.image_size)
        return self.extend_model(Pspnet, 4, self.image_size)


if __name__ == "__main__":
    images_folder_path = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/rgbImages"
    model_path = r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/model/checkpoint.model.h5"
    output_path = (
        r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/predictions"
    )
    PredictResnetUnet(
        images_folder_path=images_folder_path,
        model_path=model_path,
        output_path=output_path,
        image_size=473,
    ).predict()
