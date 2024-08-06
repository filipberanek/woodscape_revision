"""
Implemented according https://github.com/qubvel/segmentation_models.pytorch
"""

import torch
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


class Model(pl.LightningModule):

    def __init__(
        self,
        arch,
        encoder_name,
        in_channels,
        out_classes,
        loss_dict,
        optimizer,
        init_lr,
        wandb_inst,
        **kwargs,  # ,wandb_inst
    ):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Setup optimizer
        self.optimizer = optimizer
        self.init_lr = init_lr

        # Avg loss per epoch
        self.avg_epochs_loss = []

        # Logging inst
        self.wandb_inst = wandb_inst

        # for image segmentation dice loss could be the best first choice
        self.loss_dict = loss_dict
        if loss_dict["name"] == "dice_loss":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, **loss_dict["params"])
        elif loss_dict["name"] == "torch_cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(**loss_dict["params"])
        elif loss_dict["name"] == "focal_loss":
            self.loss_fn = smp.losses.FocalLoss(mode="multiclass", **loss_dict["params"])
        elif loss_dict["name"] == "rmse":
            self.loss_fn = torch.nn.MSELoss(**loss_dict["params"])
        else:
            raise NotImplementedError("This loss is not implemented")

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        # assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        # assert mask.max() <= 1.0 and mask.min() >= 0
        logits_mask = self.forward(image)

        # for image segmentation dice loss could be the best first choice
        if self.loss_dict["name"] == "dice_loss":
            mask = mask.long()
        elif self.loss_dict["name"] == "torch_cross_entropy":
            mask = mask.long()
            mask = torch.nn.functional.one_hot(mask, 4)
            mask = mask.float()
            mask = torch.moveaxis(mask, -1, 1)
        elif self.loss_dict["name"] == "focal_loss":
            mask = mask.long()
        elif self.loss_dict["name"] == "rmse":
            mask = mask.long()
            mask = torch.nn.functional.one_hot(mask, 4)
            mask = mask.float()
            mask = torch.moveaxis(mask, -1, 1)
        else:
            raise NotImplementedError("This loss is not implemented")

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        if self.loss_dict["name"] == "rmse":
            loss = torch.sqrt(self.loss_fn(logits_mask, mask))
        else:
            loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        if (self.loss_dict["name"] != "torch_cross_entropy") and (self.loss_dict["name"] != "rmse"):
            mask = torch.nn.functional.one_hot(mask, 4)
            mask = mask.float()
            mask = torch.moveaxis(mask, -1, 1)
        accuracy = torch.sum(prob_mask.argmax(1) == mask.argmax(1)).item() / len(mask.argmax(1).flatten())

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multiclass", num_classes=4)

        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": accuracy}

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        accuracy = np.mean([x["accuracy"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_accuracy": accuracy,
        }
        for metric in metrics.items():
            self.wandb_inst.log({metric[0]: metric[1]})
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="max", patience=5, eps=0.2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_accuracy",
            },
        }


def run_training(
    architecture,
    encoder,
    epochs,
    dataset_root_path: pathlib.Path,
    train_txt_filename,
    val_txt_filename,
    model_output: pathlib.Path,
    loss,
    wandb_inst,
    resume_training_model="",
):
    """
    Available architectures are:
    ['unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan']
    """
    """
    Available encoders are: 
    ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 
    'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 
    'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 
    'vgg19', 'vgg19_bn', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 
    'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 
    'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 
    'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception', 'timm-efficientnet-b0', 
    'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 
    'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 
    'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 
    'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4', 
    'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 
    'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 
    'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 
    'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 
    'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 
    'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 
    'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 
    'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 
    'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 
    'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100', 
    'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100', 
    'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l', 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 
    'mit_b5', 'mobileone_s0', 'mobileone_s1', 'mobileone_s2', 'mobileone_s3', 'mobileone_s4']
    """
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
    # Model definition
    if resume_training_model == "":
        model = Model(
            arch=architecture,
            encoder_name=encoder,
            in_channels=3,
            out_classes=4,
            loss_dict={"name": loss, "params": {}},
            optimizer="adam",
            init_lr=0.01,
            wandb_inst=wandb_inst,
        )
    else:
        model = Model.load_from_checkpoint(
            resume_training_model,
            arch=architecture,
            encoder_name=encoder,
            in_channels=3,
            out_classes=4,
            loss_dict={"name": loss, "params": {}},
            optimizer="adam",
            init_lr=0.01,
            wandb_inst=wandb_inst,
        )
    # Trainer callbacks definition
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_output / f"{architecture}_{encoder}",
        filename=f"{architecture}_{encoder}",
        monitor="valid_accuracy",
        save_top_k=1,
        mode="max",
    )
    early_stop = EarlyStopping(monitor="valid_accuracy", min_delta=0.00, patience=10, verbose=True, mode="max")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(model_output / f"{architecture}_{encoder}" / "logs"))
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
    arch = "deeplabv3"
    encoder = "resnet34"
    epochs = 50
    wandb_project = "Occlusion_detector"
    wandb.login()
    wandb.init(project=wandb_project)
    run_training(
        architecture=arch,
        encoder=encoder,
        epochs=epochs,
        dataset_root_path=train_dataset_root,
        train_txt_filename=train_txt_filename,
        val_txt_filename=val_txt_filename,
        model_output=model_output,
        loss="rmse",
        wandb_inst=wandb,
    )
    print()
