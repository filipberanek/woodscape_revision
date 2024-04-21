import sys
import os
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_segmentation.models import pspnet
from training_base import TrainingBase
from tensorflow.keras.models import Model
import wandb


class PredefinedModel(TrainingBase):
    def __init__(
        self,
        train_txt_file_with_inputs,
        val_txt_file_with_inputs,
        dataset_root,
        model_output_path,
        learning_rate,
        number_of_epochs,
        batch_size,
        wandb_inst,
        width=473,
        height=473,
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
            wandb_inst,
        )

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
        print(model.summary())
        return model

    def model_load(self):
        # input_height=473, input_width=473
        Pspnet = pspnet.pspnet_101(4, self.height, self.width)
        return self.extend_model(Pspnet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train psnet")
    parser.add_argument(
        "--train_txt_file_with_inputs",
        help="Full file path to the txt file with list of training file names",
        default=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/train_all_files.txt",
    )
    parser.add_argument(
        "--val_txt_file_with_inputs",
        help="Full file path to the txt file with list of training file names",
        default=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/val_all_files.txt",
    )
    parser.add_argument(
        "--dataset_root",
        help="Size of the image. Ration is 1:1, so provided value" "resolution should be as image_size:image_size",
        default=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling",
    )
    parser.add_argument(
        "--model_path",
        help="Folder path, where sould be stored models",
        default=r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/model/psp101",
    )
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Model learning rate")
    parser.add_argument("--number_of_epochs", default=1, help="Number of epochs for model training")
    parser.add_argument(
        "--width",
        type=int,
        default=473,
        help="Image and label width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=473,
        help="Image and label height",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--wandb_project", type=str, default="Occlusion_detector", help="Wandb project name")
    args = parser.parse_args()
    # Initiate model with parameters
    wandb.login()
    wandb.init(project=args.wandb_project)
    # Initiate model with parameters
    model = PredefinedModel(
        train_txt_file_with_inputs=args.train_txt_file_with_inputs,
        val_txt_file_with_inputs=args.val_txt_file_with_inputs,
        dataset_root=args.dataset_root,
        model_output_path=args.model_path,
        learning_rate=args.learning_rate,
        number_of_epochs=args.number_of_epochs,
        width=args.width,
        height=args.height,
        batch_size=args.config.batch_size,
        wandb_inst=wandb,
    )
    model.train_model()
