"""
Implemented according https://github.com/qubvel/segmentation_models.pytorch
"""

import pathlib
import sys
import os
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, str(pathlib.Path(dir_path).parent))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm


from train.segformer_l_base_train import Model

from transformers import SegformerFeatureExtractor


class Dataset(BaseDataset):
    CLASSES = [
        "clean",
        "transparent",
        "semi-transparent",
        "opaque",
    ]

    def __init__(
        self,
        images_path: pathlib.Path,
        labels_path: pathlib.Path,
        feature_extractor,
    ):
        self.list_of_images = list(images_path.rglob("*.png"))
        self.list_of_labels = list(labels_path.rglob("*.png"))
        self.feature_extractor = feature_extractor

    def __getitem__(self, i):
        # read data
        img = np.array(Image.open(self.list_of_images[i]).resize((512, 512)))
        lbl = np.array(Image.open(self.list_of_labels[i]).resize((512, 512)))
        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(img, lbl, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs, self.list_of_images[i].name

    def __len__(self):
        return len(self.list_of_images)


if __name__ == "__main__":
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
    parser = argparse.ArgumentParser(description="Predict with pytorch segmentation network")
    parser.add_argument(
        "--images_input_path",
        help="Full file path to the txt file with list of training file names",
        default=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/rgbImages",
    )
    parser.add_argument(
        "--labels_input_path",
        help="Full file path to the txt file with list of training file names",
        default=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/gtLabels",
    )
    parser.add_argument(
        "--model_path",
        help="Full file path to the txt file with list of training file names",
        default=r"/home/fberanek/Desktop/learning/my_articles/test/pytorch_networks/nvidia/segformer-b0-finetuned-ade-512-512_val_loss_0755.ckpt",
    )
    parser.add_argument("--wandb_project", type=str, default="Occlusion_detector", help="Wandb project name")
    args = parser.parse_args()
    images_input_path = pathlib.Path(args.images_input_path)
    model_path = pathlib.Path(args.model_path)
    # Define classes
    id2label = {"Clean": 0, "Transparent": 1, "Semi-Transparent": 2, "Opaque": 3}
    # Initiate wand
    wandb.login()
    wandb.init(project=args.wandb_project)  # args.wandb_project)
    # Model definition
    model = Model.load_from_checkpoint(
        checkpoint_path=model_path,
        id2label=id2label,
        train_dataloader="",
        val_dataloader="",
        metrics_interval=10,
        model_name="nvidia/segformer-b0-finetuned-ade-512-512",
    )

    # Create output folder
    output_path = model_path.parent.parent.parent / "predictions"
    output_path.mkdir(parents=True, exist_ok=True)

    # Instantiate network
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = 512  # Maybe put there 512
    # Get data loaders
    # Datalaoder
    test_dataset = Dataset(
        images_path=pathlib.Path(args.images_input_path),
        labels_path=pathlib.Path(args.labels_input_path),
        feature_extractor=feature_extractor,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    for batch, filename in test_loader:
        loss, logits = model(batch["pixel_values"], batch["labels"])
        upsampled_logits = nn.functional.interpolate(
            logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        predicted = predicted.squeeze()
        predicted = predicted.detach().cpu().numpy()
        predicted = predicted.astype(np.uint8)
        Image.fromarray(predicted).save(output_path / filename[0])
