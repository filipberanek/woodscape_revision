import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_segmentation.models import unet

from training_base import TrainingBase


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

    def model_load(self):
        # Library is not changing number of params, so keep 3 as encoder level. 
        Unet = unet.resnet50_unet(4, self.height, self.width, self.encoder_level)
        return self.extend_model(Unet, 4)


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
        encoder_level=3,
    )
    model.train_model()
