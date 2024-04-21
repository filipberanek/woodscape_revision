import sys
import os
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from keras_segmentation.models import unet
from training_base import TrainingBase
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
        width=512,
        height=512,
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

    def model_load(self):
        # Library is not changing number of params, so keep 3 as encoder level.
        Unet = unet.resnet50_unet(4, self.height, self.width)
        return self.extend_model(Unet, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train resnet50 unet")
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
        default=r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/model/resnet50_unet",
    )
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Model learning rate")
    parser.add_argument("--number_of_epochs", default=1, help="Number of epochs for model training")
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image and label width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image and label height",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--wandb_project", type=str, default="Occlusion_detector", help="Wandb project name")
    args = parser.parse_args()
    # Initiate model with parameters
    wandb.login()
    wandb.init(project=args.wandb_project)

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
