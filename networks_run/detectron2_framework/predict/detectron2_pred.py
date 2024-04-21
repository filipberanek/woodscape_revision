from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import warnings
import GPUtil as GPU
import os
import humanize
import psutil
import torch
import cv2
import json
import numpy as np
import pathlib
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_gpu_status():
    print(torch.__version__, torch.cuda.is_available())

    GPUs = GPU.getGPUs()
    # XXX: only one GPU on Colab and isn’t guaranteed
    gpu = GPUs[0]

    def printm():
        process = psutil.Process(os.getpid())
        print(
            "Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
            " | Proc size: " + humanize.naturalsize(process.memory_info().rss),
        )
        print(
            "GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(
                gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal
            )
        )

    printm()


def get_dict(dict_file_path):
    with open(dict_file_path) as f:
        imgs_anns = json.load(f)
    return imgs_anns


def predict(images_folder_path, model_path, output_path):
    # Create output folder
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # Setup loger
    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/semantic_R_50_FPN_1x.yaml"))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 4
    cfg.MODEL.WEIGHTS = model_path  # path to the model we just trained
    predictor = DefaultPredictor(cfg)
    lof = list(pathlib.Path(images_folder_path).glob("*.png"))
    for file_path in tqdm(lof):
        im = cv2.imread(str(file_path))
        outputs = predictor(im)
        outputs = torch.argmax(outputs["sem_seg"].to("cpu"), dim=0)
        img = Image.fromarray(outputs.numpy().astype(np.uint8))
        img.save(str(output_path / file_path.name))


if __name__ == "__main__":
    get_gpu_status()
    images_folder_path = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/rgbImages"
    model_path = r"/home/fberanek/Desktop/learning/my_articles/outputs/detectron2/detectron2_correct_clear_strict_files/model/model_final.pth"
    output_path = (
        r"/home/fberanek/Desktop/learning/my_articles/outputs/detectron2/detectron2_correct_clear_strict_files/predictions"
    )
    predict(
        images_folder_path=images_folder_path,
        model_path=model_path,
        output_path=output_path,
    )
