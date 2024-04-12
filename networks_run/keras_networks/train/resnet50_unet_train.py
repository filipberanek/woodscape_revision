import sys
import os
import argparse

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
        wandb_inst,
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
    parser = argparse.ArgumentParser(description="Train Deep labV3")
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
    args = parser.parse_args()
    # Initiate model with parameters
    config = dict(
        epochs=2,
        learning_rate=0.001,
        width=512,
        height=512,
        batch_size=[
            2,
            3,
        ],
    )
    sweep_config = {"method": "random"}
    metric = {"name": "val_accuracy", "goal": "maximize"}
    sweep_config["metric"] = metric
    parameters_dict = {
        "optimizer": {"values": ["adam", "sgd"]},
    }
    sweep_config["parameters"] = parameters_dict
    parameters_dict.update(
        {
            "learning_rate": {
                # a flat distribution between 0 and 0.1
                "distribution": "uniform",
                "min": 0,
                "max": 0.1,
            },
            "batch_size": {
                # integers between 32 and 256
                # with evenly-distributed logarithms
                "distribution": "q_log_uniform_values",
                "q": 8,
                "min": 2,
                "max": 16,
            },
        }
    )
    parameters_dict.update(
        {
            "train_txt_file_with_inputs": r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/train_all_files.txt",
            "val_txt_file_with_inputs": r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/val_all_files.txt",
            "dataset_root": r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling",
            "model_path": r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/model/resnet50_unet",
            "width": 512,
            "height": 512,
            "number_of_epochs": 1000,
        }
    )
    import wandb

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    with wandb.init(config=config):
        model = PredefinedModel(
            train_txt_file_with_inputs=args.train_txt_file_with_inputs,
            val_txt_file_with_inputs=args.val_txt_file_with_inputs,
            dataset_root=args.dataset_root,
            model_output_path=args.model_path,
            learning_rate=wandb.config.learning_rate,
            number_of_epochs=wandb.config.number_of_epochs,
            width=wandb.config.width,
            height=wandb.config.height,
            batch_size=wandb.config.batch_size,
            wandb_inst=wandb,
        )
        model.train_model()
