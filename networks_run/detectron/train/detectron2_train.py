from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator, SemSegEvaluator, DatasetEvaluators
import warnings
import GPUtil as GPU
import os
import humanize
import psutil
import torch
import json
import numpy as np
import time
import datetime
import logging
import random

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


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        print(f"mean_loss {mean_loss}")
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        print(f"metrics_dict {metrics_dict}")
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v) for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class CustomTrainer(DefaultTrainer):
    """
    Custom Trainer deriving from the "DefaultTrainer"

    Overloads build_hooks to add a hook to calculate loss on the test set during training.
    """

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                10,  # Frequency of calculation - every 100 iterations here
                self.model,
                build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True)),
            ),
        )

        return hooks


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
        sem_seg_evaluator = SemSegEvaluator(dataset_name=dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR)

        evaluator_list = [coco_evaluator, sem_seg_evaluator]

        return DatasetEvaluators(evaluator_list)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class Detectron2Trainer:
    def __init__(self, root, model_output_path, dataset, max_iteration, learning_rate) -> None:
        self.root = root
        self.model_output_path = model_output_path
        self.dataset = dataset
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate

    def train(self):
        # Setup loger
        setup_logger()

        for d in self.dataset:
            DatasetCatalog.register(f"mud_{d}", lambda d=d: get_dict(f"{self.root}/{d}.json"))
            MetadataCatalog.get(f"mud_{d}").set(
                thing_classes=["clear", "transparent", "semi-transparent", "opaque"],
                stuff_classes=["clear", "transparent", "semi-transparent", "opaque"],
                ignore_label=False,
            )
        set_seed()
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("Misc/semantic_R_50_FPN_1x.yaml"))
        cfg.DATASETS.TRAIN = (f"mud_{self.dataset[0]}",)
        cfg.DATASETS.TEST = (f"mud_{self.dataset[1]}",)
        if int(self.max_iteration / 100) > 10:
            cfg.TEST.EVAL_PERIOD = int(self.max_iteration / 100)
        cfg.DATALOADER.NUM_WORKERS = 2
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/semantic_R_50_FPN_1x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 5
        cfg.SOLVER.BASE_LR = self.learning_rate  # 0.00025  # pick a good LR 0.00025
        cfg.SOLVER.MAX_ITER = self.max_iteration
        cfg.SOLVER.WARMUP_ITERS = int(0.01 * cfg.SOLVER.MAX_ITER)
        cfg.SOLVER.LR_FACTOR_FUNC = 0.1  # TODO remove
        cfg.SOLVER.GAMMA = 0.5
        cfg.SOLVER.STEPS = [int(x) for x in np.linspace(0, cfg.SOLVER.MAX_ITER, 11)][1:-1]
        # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # 128
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 4
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        cfg.OUTPUT_DIR = self.model_output_path
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        # trainer = DefaultTrainer(cfg=cfg)
        # trainer = CustomTrainer(cfg=cfg)
        trainer = MyTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()


if __name__ == "__main__":
    get_gpu_status()
    root = r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train"
    model_output_path = r"/home/fberanek/Desktop/learning/my_articles/outputs/detectron2"
    dataset = ["train_all_files", "val_all_files"]
    Detectron2Trainer(root, model_output_path, dataset, 50, 0.0001).train()
