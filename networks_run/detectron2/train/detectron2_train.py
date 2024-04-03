from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
import warnings
import random
import GPUtil as GPU
import os
import humanize
import psutil
import torch
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
warnings.filterwarnings('ignore')


def get_gpu_status():
    print(torch.__version__, torch.cuda.is_available())

    GPUs = GPU.getGPUs()
    # XXX: only one GPU on Colab and isn’t guaranteed
    gpu = GPUs[0]

    def printm():
        process = psutil.Process(os.getpid())
        print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
              " | Proc size: " + humanize.naturalsize(process.memory_info().rss))
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(
            gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal))

    printm()


def get_dict(dict_file_path):
    with open(dict_file_path) as f:
        imgs_anns = json.load(f)
    return imgs_anns


def train(root, model_output_path, dataset):
    # Setup loger
    setup_logger()

    for d in [dataset]:
        DatasetCatalog.register(f"mud_{dataset}", lambda d=d: get_dict(f"{root}/{d}.json"))
        MetadataCatalog.get(f"mud_{dataset}").set(thing_classes=["clear", "transparent", "semi-transparent", "opaque"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/semantic_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = (f"mud_{dataset}",)
    cfg.DATASETS.TEST = (f"mud_{dataset}",)
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/semantic_R_50_FPN_1x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 5
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR 0.00025
    cfg.SOLVER.MAX_ITER = 50
    cfg.SOLVER.WARMUP_ITERS = int(0.01*cfg.SOLVER.MAX_ITER)
    # cfg.SOLVER.LR_FACTOR_FUNC = 0.01
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = [int(x) for x in np.linspace(0, cfg.SOLVER.MAX_ITER, 11)][1:-1]
    # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # 128
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = model_output_path
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    get_gpu_status()
    root = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train"
    model_output_path = r"/home/fberanek/Desktop/learning/my_articles/outputs/detectron2"
    dataset = "train_correct"
    train(root, model_output_path, dataset)
