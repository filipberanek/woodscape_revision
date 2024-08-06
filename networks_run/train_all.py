import argparse

import pathlib
import wandb
import pandas as pd

from keras_networks.train import (
    psp_101_train,
    resnet50_pspnet_train,
    resnet50_segnet_train,
    resnet50_unet_train,
)
from detectron2_framework.train.detectron2_train import Detectron2Trainer
from pytorch_networks.train.pytorch_l_base_train import run_training as PytorchNetworksRun

# Define script variables
LIST_OF_FILES_FOR_TRAIN_VAL = [
    "all_files",
    # "correct_files",
    # "correct_clear_files",
    # "correct_clear_strict_files",
]  # "all_files", "correct_files", "correct_clear_files", "correct_clear_strict_files"
LIST_OF_NETWORKS_KERAS = [
    # resnet50_unet_train,
    # resnet50_pspnet_train,
    resnet50_segnet_train,
]  # Detectron2Trainer
LIST_OF_NETWORKS_STRINGS = [str(network_name.__name__) for network_name in LIST_OF_NETWORKS_KERAS]
DF_OF_NETWORKS = pd.DataFrame(
    [{"network_name": network_obj.__name__, "network_object": network_obj} for network_obj in LIST_OF_NETWORKS_KERAS]
)
LIST_OF_PYTORCH_NETWORKS = [
    # "unet",
    # "unetplusplus",
    # "deeplabv3",
    # "deeplabv3plus",
    "fpn",
    # "pspnet",
    # "manet",
    # "linknet",
    # "pan",
]
LIST_OF_PYTORCH_ENCODERS = ["resnet152"]  # "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
LIST_OF_LOSSES = [
    # "focal_loss",
    # "dice_loss",
    # "rmse",
    "torch_cross_entropy",
]  # "focal_loss", "dice_loss", "rmse", "torch_cross_entropy"
KERAS_START_LR = 0.01
NUMBER_OF_EPOCHS = 400
RUN_KERAS = False
RUN_DETECTRON = False
RUN_PYTORCH = True
# log into W&
wandb.login()


def objective_keras(config, wandb_inst):
    network_obj = DF_OF_NETWORKS[DF_OF_NETWORKS["network_name"] == config["model"]]["network_object"].item()
    train_data_root = pathlib.Path(config["train_data_root"])
    models_output = pathlib.Path(config["models_output"])
    txt_train_file = train_data_root / "train" / f"train_{config['dataset']}.txt"
    txt_val_file = train_data_root / "train" / f"val_{config['dataset']}.txt"
    network_inst = network_obj.PredefinedModel(
        train_txt_file_with_inputs=txt_train_file,
        val_txt_file_with_inputs=txt_val_file,
        dataset_root=config["train_data_root"],
        model_output_path=models_output
        / f"{config['framework']}"
        / f"{config['model'].split('.')[-1]}_{config['dataset']}"
        / "model",
        learning_rate=config["learning_rate"],
        number_of_epochs=config["number_of_epchs"],
        batch_size=config["batch_size"],
        wandb_inst=wandb_inst,
    )
    network_inst.train_model()
    return  # Score


def main_keras():
    wandb.init(project="Occlusion_detector")
    objective_keras(wandb.config, wandb)


def objective_detectron2(config):
    train_data_root = pathlib.Path(config["train_data_root"])
    models_output = pathlib.Path(config["models_output"])
    txt_train_file = train_data_root / "train" / f"train_{config['dataset']}.json"
    txt_val_file = train_data_root / "train" / f"val_{config['dataset']}.json"
    network_inst = Detectron2Trainer(
        train_txt_file_with_inputs=txt_train_file,
        val_txt_file_with_inputs=txt_val_file,
        model_output_path=models_output
        / f"{config['framework']}"
        / f"{config['framework']}_batch_size_per_image_{config['batch_size_per_image']}_{config['dataset']}"
        / "model",
        learning_rate=config["learning_rate"],
        number_of_epochs=config["number_of_epchs"],
        images_per_batch=config["images_per_batch"],
        batch_size_per_image=config["batch_size_per_image"],
    )
    network_inst.train()
    return  # Score


def main_detectron2():
    wandb.init(project="Occlusion_detector")
    objective_detectron2(wandb.config)


def objective_pytorch(config, wandb_inst):
    train_data_root = pathlib.Path(config["train_data_root"])
    models_output_path = pathlib.Path(config["models_output"])
    txt_train_file = train_data_root / "train" / f"train_{config['dataset']}.txt"
    txt_val_file = train_data_root / "train" / f"val_{config['dataset']}.txt"
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
    wandb.init(project="Occlusion_detector")
    objective_pytorch(wandb.config, wandb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all networks")
    parser.add_argument(
        "--train_data_root",
        help="Folder path to dataset root folder",
        default=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling",
    )
    parser.add_argument(
        "--models_output",
        help="folder path, where results should be stored",
        default=r"/home/fberanek/Desktop/learning/my_articles/outputs",
    )
    args = parser.parse_args()
    if RUN_DETECTRON:
        for dataset_name in LIST_OF_FILES_FOR_TRAIN_VAL:
            # 1: Define objective/training function

            # 2: Define the search space
            """
            "batch_size_per_image": {
                # with evenly-distributed logarithms
                "distribution": "int_uniform",
                # "q": 64,
                "min": 128,
                "max": 512,
            },
            """
            sweep_configuration = {
                "method": "random",
                "name": f"Detectron2_batch_size_per_image_{dataset_name}",
                "metric": {"name": "val_accuracy", "goal": "maximize"},
                "dataset": dataset_name,
                "parameters": {
                    "images_per_batch": {"value": 8},
                    "batch_size_per_image": {"value": 128},
                    "dataset": {"value": dataset_name},
                    "train_data_root": {"value": args.train_data_root},
                    "models_output": {"value": args.models_output},
                    "learning_rate": {"value": KERAS_START_LR},
                    "number_of_epchs": {"value": NUMBER_OF_EPOCHS * 5},
                    "framework": {"value": "Detectron2"},
                },
            }

            # 3: Start the sweep
            # Run Keras
            sweep_id = wandb.sweep(sweep=sweep_configuration, project="Occlusion_detector")

            wandb.agent(
                sweep_id,
                function=main_detectron2,
                count=1,
            )
            wandb.finish()
    if RUN_KERAS:
        for network in LIST_OF_NETWORKS_STRINGS:
            for dataset_name in LIST_OF_FILES_FOR_TRAIN_VAL:
                # 1: Define objective/training function

                # 2: Define the search space
                sweep_configuration = {
                    "method": "random",
                    "name": f'{network.split(".")[0]}_{network.split(".")[-1]}_{dataset_name}',
                    "metric": {"name": "val_accuracy", "goal": "maximize"},
                    "dataset": dataset_name,
                    "parameters": {
                        "batch_size": {"value": 2},
                        "learning_rate": {"value": KERAS_START_LR},
                        "dataset": {"value": dataset_name},
                        "train_data_root": {"value": args.train_data_root},
                        "models_output": {"value": args.models_output},
                        "model": {"value": network},
                        "framework": {"value": network},
                        "number_of_epchs": {"value": NUMBER_OF_EPOCHS},
                    },
                }

                # 3: Start the sweep
                # Run Keras
                sweep_id = wandb.sweep(sweep=sweep_configuration, project="Occlusion_detector")

                wandb.agent(
                    sweep_id,
                    function=main_keras,
                    count=1,
                )
                wandb.finish()
    if RUN_PYTORCH:
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
                        # Run Keras
                        sweep_id = wandb.sweep(sweep=sweep_configuration, project="Occlusion_detector")

                        wandb.agent(
                            sweep_id,
                            function=main_pytorch,
                            count=1,
                        )
                        wandb.finish()
                        print()
