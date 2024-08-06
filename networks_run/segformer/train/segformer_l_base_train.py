"""
Implemented according https://github.com/qubvel/segmentation_models.pytorch
"""

import torch
from torch import nn
import pathlib

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import wandb
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric


class Dataset(BaseDataset):
    CLASSES = [
        "clean",
        "transparent",
        "semi-transparent",
        "opaque",
    ]

    def __init__(
        self,
        dataset_root: pathlib.Path,
        dataset_txt_file,
        feature_extractor,
    ):
        list_of_files = open(str(dataset_root / dataset_txt_file)).readlines()
        self.list_of_images = [f"{dataset_root}{row.split(',')[0].strip()}" for row in list_of_files]
        self.list_of_labels = [f"{dataset_root}{row.split(',')[1].strip()}" for row in list_of_files]
        self.feature_extractor = feature_extractor

    def __getitem__(self, i):
        # read data
        img = np.array(Image.open(self.list_of_images[i]).resize((512, 512)))
        lbl = np.array(Image.open(self.list_of_labels[i]).resize((512, 512)))
        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(img, lbl, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs

    def __len__(self):
        return len(self.list_of_images)


class Model(pl.LightningModule):

    def __init__(
        self,
        model_name,
        id2label,
        train_dataloader=None,
        val_dataloader=None,
        metrics_interval=10,
    ):
        super().__init__()
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.id2label = id2label
        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return outputs

    def training_step(self, batch, batch_nb):

        images, masks = batch["pixel_values"], batch["labels"]

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy()
        )
        if batch_nb % self.metrics_interval == 0:

            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
            )

            metrics = {"loss": loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}

            for k, v in metrics.items():
                self.log(k, v)

            return metrics
        else:
            return {"loss": loss}

    def validation_step(self, batch, batch_nb):

        images, masks = batch["pixel_values"], batch["labels"]

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy()
        )

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"val_loss": avg_val_loss, "val_mean_iou": val_mean_iou, "val_mean_accuracy": val_mean_accuracy}
        for k, v in metrics.items():
            self.log(k, v)

        return metrics

    def test_step(self, batch, batch_nb):

        images, masks = batch["pixel_values"], batch["labels"]

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy()
        )

        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss": avg_test_loss, "test_mean_iou": test_mean_iou, "test_mean_accuracy": test_mean_accuracy}

        for k, v in metrics.items():
            self.log(k, v)

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-03, eps=1e-08)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="max", patience=5, eps=0.2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mean_accuracy",
            },
        }

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


def run_training(
    model_name,
    epochs,
    image_size,
    dataset_root_path: pathlib.Path,
    train_txt_filename,
    val_txt_filename,
    model_output: pathlib.Path,
    wandb_inst,
    resume_training_model="",
):
    # Define classes
    id2label = {"Clean": 0, "Transparent": 1, "Semi-Transparent": 2, "Opaque": 3}
    # Instantiate network
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = image_size  # Maybe put there 512
    # Get data loaders
    # Datalaoder
    train_dataset = Dataset(
        dataset_root=dataset_root_path, dataset_txt_file=train_txt_filename, feature_extractor=feature_extractor
    )

    valid_dataset = Dataset(
        dataset_root=dataset_root_path, dataset_txt_file=val_txt_filename, feature_extractor=feature_extractor
    )
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=1)
    # Model definition
    if resume_training_model == "":
        model = Model(
            id2label=id2label,
            train_dataloader=train_loader,
            val_dataloader=valid_loader,
            metrics_interval=10,
            model_name=model_name,
        )
    else:
        model = Model.load_from_checkpoint(
            checkpoint_path=resume_training_model,
            id2label=id2label,
            train_dataloader=train_loader,
            val_dataloader=valid_loader,
            metrics_interval=10,
            model_name=model_name,
        )
    # Trainer callbacks definition
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_output,
        filename=f"{model_name}",
        monitor="val_mean_accuracy",
        save_top_k=1,
        mode="max",
    )
    early_stop = EarlyStopping(monitor="val_mean_accuracy", min_delta=0.00, patience=20, verbose=True, mode="max")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(model_output / f"{model_name}" / "logs"))
    # Trainer definition
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop],
        num_sanity_val_steps=0,
        logger=tb_logger,
    )
    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )


if __name__ == "__main__":
    # Parameters
    train_dataset_root = pathlib.Path(r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train")
    train_txt_filename = "train_correct_clear_strict_files.txt"
    val_txt_filename = "val_correct_clear_strict_files.txt"
    model_output = pathlib.Path(r"/home/fberanek/Desktop/learning/my_articles/test/pytorch_networks")
    epochs = 50
    wandb_project = "Occlusion_detector"
    wandb.login()
    wandb.init(project=wandb_project)
    run_training(
        model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        epochs=epochs,
        image_size=512,
        dataset_root_path=train_dataset_root,
        train_txt_filename=train_txt_filename,
        val_txt_filename=val_txt_filename,
        model_output=model_output,
        wandb_inst=wandb,
        resume_training_model=r"/home/fberanek/Desktop/learning/my_articles/test/pytorch_networks/nvidia/segformer-b0-finetuned-ade-512-512_val_loss_0755.ckpt",
    )
