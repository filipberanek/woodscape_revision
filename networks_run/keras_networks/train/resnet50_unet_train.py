import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_segmentation.models import unet

from training_base import TrainingBase


class PredefinedModel(TrainingBase):
    def __init__(
        self,
        txt_file_with_inputs,
        dataset_root,
        model_output_path,
        val_coeficient,
        learning_rate,
        number_of_epochs,
        image_size,
        batch_size,
        encoder_level,
    ) -> object:
        super().__init__(
            txt_file_with_inputs,
            dataset_root,
            model_output_path,
            val_coeficient,
            learning_rate,
            number_of_epochs,
            image_size,
            batch_size,
        )
        self.encoder_level = encoder_level

    def model_load(self):
        Unet = unet.resnet50_unet(4, self.image_size, self.image_size, 3)
        return self.extend_model(Unet, 4, self.image_size)


if __name__ == "__main__":
    # Initiate model with parameters
    model = PredefinedModel(
        txt_file_with_inputs=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/correct.txt",
        dataset_root=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling",
        model_output_path=r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/model",
        val_coeficient=0.1,
        learning_rate=0.0001,
        number_of_epochs=1,
        image_size=512,
        batch_size=2,
        encoder_level=3,
    )
    model.train_model()
