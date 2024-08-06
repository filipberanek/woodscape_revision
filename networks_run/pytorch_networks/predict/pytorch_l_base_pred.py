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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm


from train.pytorch_l_base_train import Model


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
    ):
        list_of_files = pathlib.Path(dataset_root)
        self.list_of_images = list(list_of_files.rglob("*.png"))

    def __getitem__(self, i):
        # read data
        img = np.array(Image.open(self.list_of_images[i]).resize((512, 512)))
        img = torch.from_numpy(img)
        img = torch.moveaxis(img, -1, 0) / 255.0  # .float()
        return img, [str(self.list_of_images[i])]

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
        "--images_input_path", help="Full file path to the txt file with list of training file names", required=True
    )
    parser.add_argument(
        "--model_path", help="Full file path to the txt file with list of training file names", required=True
    )
    parser.add_argument("--network_architecture", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True, help="Wandb project name")
    args = parser.parse_args()
    images_input_path = pathlib.Path(args.images_input_path)
    model_path = pathlib.Path(args.model_path)

    # Initiate wand
    wandb.login()
    wandb.init(project=args.wandb_project)  # args.wandb_project)
    # Model definition
    model = Model.load_from_checkpoint(
        args.model_path,
        arch=args.network_architecture,
        encoder_name=args.encoder,
        in_channels=3,
        out_classes=4,
        loss_dict={"name": "torch_cross_entropy", "params": {}},
        optimizer="adam",
        init_lr=0.01,
        wandb_inst=wandb,
    )
    # Create output folder
    output_path = model_path.parent.parent.parent / "predictions"
    output_path.mkdir(parents=True, exist_ok=True)

    test_dataset = Dataset(
        dataset_root=images_input_path,
    )

    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=1)

    for batch, filenames in tqdm(test_loader):
        pred = model(batch)
        if len(pred.shape) == 4:
            for pred_single, filename in zip(pred, filenames[0]):
                pred_single = pred_single.squeeze()
                pred_single = torch.argmax(pred_single, 0).detach().cpu().numpy()
                pred_single = pred_single.astype(np.uint8)
                Image.fromarray(pred_single).save(output_path / pathlib.Path(filename).name)
