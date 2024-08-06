import sys
import os
import pathlib

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_pred_base import PredictBase
from keras_segmentation.models import unet


class PredictResnetUnet(PredictBase):
    def load_model(self):
        Unet = unet.resnet50_unet(4, self.height, self.width, 3)
        return self.extend_model(Unet, 4)


if __name__ == "__main__":
    images_folder_path = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/rgbImages"
    model_path_root = pathlib.Path(
        r"/home/fberanek/Desktop/learning/my_articles/outputs/keras_networks/resnet50_unet_train_correct_clear_strict_files"
    )
    model_path = model_path_root / "model" / "checkpoint.model.h5"
    output_path = model_path_root / "predictions"
    PredictResnetUnet(
        images_folder_path=images_folder_path,
        model_path=model_path,
        output_path=output_path,
        width=512,
        height=512,
    ).predict()
