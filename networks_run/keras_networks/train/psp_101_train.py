import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_segmentation.models import pspnet

from training_base import TrainingBase

from tensorflow.keras.models import Model


class PredefinedModel(TrainingBase):
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

    def model_load(self):
        # input_height=473, input_width=473
        Pspnet = pspnet.pspnet_101(4, self.image_size, self.image_size)
        return self.extend_model(Pspnet, 4, self.image_size)


if __name__ == "__main__":
    # Initiate model with parameters
    model = PredefinedModel(
        txt_file_with_inputs=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/correct.txt",
        dataset_root=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling",
        model_output_path=r"/home/fberanek/Desktop/learning/my_articles/outputs",
        val_coeficient=0.1,
        learning_rate=0.0001,
        number_of_epochs=1,
        image_size=473,
        batch_size=2,
    )
    model.train_model()
