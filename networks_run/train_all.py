import argparse

import pathlib
import wandb
import pandas as pd

from pytorch_networks.train.pytorch_l_base_train import run_training as PytorchNetworksRun

# Define script variables
LIST_OF_FILES_FOR_TRAIN_VAL = [
    "all_files",
    "correct_files",
    "correct_clear_files",
    "correct_clear_strict_files",
]
LIST_OF_PYTORCH_NETWORKS = [
    "unet",
    "unetplusplus",
    "deeplabv3",
    "deeplabv3plus",
    "fpn",
    "pspnet",
    "manet",
    "linknet",
    "pan",
]
LIST_OF_PYTORCH_ENCODERS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
LIST_OF_LOSSES = [
    "focal_loss",
    "dice_loss",
    "rmse",
    "torch_cross_entropy",
]
KERAS_START_LR = 0.01
NUMBER_OF_EPOCHS = 400
WANDB_PROJECT = "Occlusion_detector"
# log into W&
wandb.login()


def objective_pytorch(config, wandb_inst):
    train_data_root = pathlib.Path(config["train_data_root"])
    models_output_path = pathlib.Path(config["models_output"])
    txt_train_file = f"train_{config['dataset']}.txt"
    txt_val_file = f"val_{config['dataset']}.txt"
    PytorchNetworksRun(
        architecture=config["model"],
        encoder=config["encoder"],
        epochs=config["number_of_epchs"],
        dataset_root_path=train_data_root / "train",
        train_txt_filename=txt_train_file,
        val_txt_filename=txt_val_file,
        loss=config["loss"],
        model_output=models_output_path
        / config["framework"]
        / f"{config['model']}_{config['encoder']}_{config['loss']}_{config['dataset']}"
        / "model",
        wandb_inst=wandb_inst,
    )
    return  # Score


def main_pytorch():
    wandb.init(project=WANDB_PROJECT)
    objective_pytorch(wandb.config, wandb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all networks")
    parser.add_argument(
        "--train_data_root",
        help="Folder path to dataset root folder",
        default=r"./woodscape_preprocessed",
    )
    parser.add_argument(
        "--models_output",
        help="folder path, where results should be stored",
        default=r"./model_outputs",
    )
    args = parser.parse_args()
    for network in LIST_OF_PYTORCH_NETWORKS:
        for encoder in LIST_OF_PYTORCH_ENCODERS:
            for dataset_name in LIST_OF_FILES_FOR_TRAIN_VAL:
                for loss in LIST_OF_LOSSES:
                    # 1: Define objective/training function

                    # 2: Define the search space
                    sweep_configuration = {
                        "method": "random",
                        "name": f"pytorch_{network}_{encoder}_{dataset_name}",
                        "metric": {"name": "valid_accuracy", "goal": "maximize"},
                        "dataset": dataset_name,
                        "parameters": {
                            "batch_size": {"value": 2},
                            "learning_rate": {"value": KERAS_START_LR},
                            "dataset": {"value": dataset_name},
                            "train_data_root": {"value": args.train_data_root},
                            "models_output": {"value": args.models_output},
                            "model": {"value": network},
                            "framework": {"value": "pytorch_networks"},
                            "number_of_epchs": {"value": NUMBER_OF_EPOCHS},
                            "encoder": {"value": encoder},
                            "loss": {"value": loss},
                        },
                    }

                    # 3: Start the sweep
                    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Occlusion_detector")

                    wandb.agent(
                        sweep_id,
                        function=main_pytorch,
                        count=1,
                    )
                    wandb.finish()
