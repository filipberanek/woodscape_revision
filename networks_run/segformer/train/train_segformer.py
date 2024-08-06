from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import pathlib
import numpy as np
import wandb

"""
This is implementation of segformer according to the: 
https://blog.roboflow.com/how-to-train-segformer-on-a-custom-dataset-with-pytorch-lightning/
"""


class Dataset(Dataset):
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
    ):
        list_of_files = open(str(dataset_root / dataset_txt_file)).readlines()
        self.list_of_images = [f"{dataset_root}{row.split(',')[0].strip()}" for row in list_of_files]
        self.list_of_labels = [f"{dataset_root}{row.split(',')[1].strip()}" for row in list_of_files]

    def __getitem__(self, i):
        # read data
        img = np.array(Image.open(self.list_of_images[i]).resize((512, 512)))
        lbl = np.array(Image.open(self.list_of_labels[i]).resize((512, 512)))
        img = torch.from_numpy(img)
        img = torch.moveaxis(img, -1, 0) / 255.0  # .float()
        lbl = torch.from_numpy(lbl)

        return img, lbl

    def __len__(self):
        return len(self.list_of_images)


class SegformerFinetuner(pl.LightningModule):

    def __init__(
        self,
        model_name,
        id2label,
        train_dataloader=None,
        val_dataloader=None,
        metrics_interval=100,
    ):
        super(SegformerFinetuner, self).__init__()
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
        """Make a forward step = make predictions
        Args:
            images (tensor): Tensor of rgb image batch
            masks (tensor): Tensor of label batch
        Returns:
            tensor: Tensoe with predictions
        """
        outputs = self.model(pixel_values=images.to(torch.float), labels=masks.to(torch.long))
        return outputs

    def training_step(self, batch, batch_nb):
        """Do a training step that consists of calculating metrics and returning it
        Args:
            batch (tensor): batch of predictions and labels
            batch_nb (tensor): batch of tensor values
        Returns:
            dict: Dictionary with metrics
        """
        images, masks = batch

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
        """Run validation steps
        Args:
            batch (tensor): batch of predictions and labels
            batch_nb (tensor): batch of tensor values
        Returns:
            dict: Dictionary with metrics
        """
        images, masks = batch

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
        """Calucalte metrices for validation epoch end
        Args:
            outputs (tensor): predictions
        Returns:
            dict: Dictionary with metrices
        """
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
        """Run test steps
        Args:
            batch (tensor): batch of predictions and labels
            batch_nb (tensor): batch of tensor values
        Returns:
            dict: Dictionary with metrics
        """
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
        """Calucalte metrices for test epoch end
        Args:
            outputs (tensor): predictions
        Returns:
            dict: Dictionary with metrices
        """
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
        """Set of optimizer
        Returns:
            object: Defined optimizer
        """
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        """Get data loader for train
        Returns:
            DataLoader: Dataloader of data for train
        """
        return self.train_dl

    def val_dataloader(self):
        """Get data loader for val
        Returns:
            DataLoader: Dataloader of data for val
        """
        return self.val_dl


def run_training(
    model_name,
    epochs,
    image_size,
    dataset_root_path: pathlib.Path,
    train_txt_filename,
    val_txt_filename,
    model_output: pathlib.Path,
    wandb_inst,
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
        dataset_root=dataset_root_path,
        dataset_txt_file=train_txt_filename,
    )

    valid_dataset = Dataset(
        dataset_root=dataset_root_path,
        dataset_txt_file=val_txt_filename,
    )
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=1)

    segformer_finetuner = SegformerFinetuner(
        id2label=id2label,
        train_dataloader=train_loader,
        val_dataloader=valid_loader,
        metrics_interval=10,
        model_name=model_name,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_output,
        filename="{args.model_output_filename}_{epoch}_{val_loss:.2f}",
        monitor="val_loss",
    )
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=epochs,
        val_check_interval=len(valid_loader),
    )
    trainer.fit(segformer_finetuner)


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
    )
    print()


"""
# Params
parser = argparse.ArgumentParser(description="Train Deep labV3")
parser.add_argument(
    "--image_path",
    help="File path to images",
    default=r"P:\09_OE_5_DSF\DSF_ValPlat\gte\mud_detection\Train\rgbImages_full",
    type=str,
)
parser.add_argument(
    "--label_path",
    help="File path to mask labels",
    default=r"P:\09_OE_5_DSF\DSF_ValPlat\gte\mud_detection\Train\rgbLabels_ready_full",
    type=str,
)
parser.add_argument(
    "--config_file_path",
    help="Config file path with classes description",
    default=r"P:\09_OE_5_DSF\DSF_ValPlat\gte\mud_detection\config.json",
    type=str,
)
parser.add_argument("--image_size", help="Image is 1:1 so image_size:image_size", type=int, default=512)
parser.add_argument("--model_weights", help="Model weights so training can resume", type=str, default=r"")
parser.add_argument(
    "--model_output_path",
    help="Folder path, where sould be stored models",
    default=r"P:\09_OE_5_DSF\DSF_ValPlat\gte\mud_detection\Train\models\segformer_debug",
    type=str,
)
parser.add_argument(
    "--model_output_filename",
    help='Name will be in the end model_output_filename"+"_{epoch}.h5',
    default="segformer",
)
parser.add_argument("--number_of_epochs", help="Folder path, where should be stored models", type=int, default=10)
parser.add_argument("--batch_size", help="Batch size used for training", type=int, default=8)
parser.add_argument("--num_workers", help="Batch size used for training", type=int, default=2)
parser.add_argument("--train_coeficient", help="Coeficient of training size", type=float, default=0.85)
parser.add_argument("--val_coeficient", help="Coeficient of training size", type=float, default=0.1)
parser.add_argument(
    "--model_name",
    help="String of pretrained model according to https://huggingface.co/models",
    type=str,
    default="nvidia/segformer-b0-finetuned-ade-512-512",
)
"""
