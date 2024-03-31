from pathlib import Path

from keras_segmentation.models import pspnet

from networks_training.keras_networks.training_base import TrainingBase

import tensorflow as tf


class PredefinedModel(TrainingBase):
    def model_load(self):
        Pspnet = pspnet.pspnet_101(4, self.image_size, self.image_size)
        return self.extend_model(Pspnet, 4, self.image_size)


if __name__ == "__main__":
    # Initiate model with parameters
    model = PredefinedModel(
        txt_file_with_inputs=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/correct.txt",
        dataset_root=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling",
        model_output_path=r"/home/fberanek/Desktop/learning/my_articles/outputs",
        model_filename="soiling",
        val_coeficient=0.1,
        learning_rate=0.0001,
        number_of_epochs=1,
        image_size=512,
        batch_size=2)
    model.train_model()
