import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_pred_base import PredictBase
from keras_segmentation.models import segnet


class PredictResnetUnet(PredictBase):
    def load_model(self):
        Segnet = segnet.resnet50_segnet(4, self.image_size, self.image_size, 3)
        return self.extend_model(Segnet, 4, self.image_size)


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
        image_size=512,
    ).predict()
