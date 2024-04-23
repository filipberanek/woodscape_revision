import warnings

from detectron2.evaluation.sem_seg_evaluation import load_image_into_numpy_array

warnings.filterwarnings("ignore")
import os
import humanize
import psutil
import json
import numpy as np
import logging
import random
import argparse
import pathlib
import itertools
from collections import OrderedDict

import torch
import GPUtil as GPU

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from typing import Optional, Union
from PIL import Image
from detectron2.evaluation import SemSegEvaluator, DatasetEvaluators
import wandb


_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


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


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array


class MySemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn("SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata.")
        if ignore_label is not None:
            self._logger.warn("SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata.")
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )
        self.output_stats = {"Correct": 0, "Total": 0}
        wandb.login()
        wandb.init(project="Occlusion_detector")
        self.wandb_inst = wandb

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        val_accuracy = {"val_ccuracy": self.output_stats["Correct"] / self.output_stats["Total"]}
        self._logger.info(val_accuracy)
        self.wandb_inst.log(val_accuracy)
        return results

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)

            self.output_stats["Correct"] += np.sum(np.array(output) == gt)
            self.output_stats["Total"] += len(output.reshape(-1))

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        sem_seg_evaluator = MySemSegEvaluator(dataset_name=dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR)

        # sem_seg_evaluator = SemSegEvaluator(dataset_name=dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR)

        evaluator_list = [sem_seg_evaluator]

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
    def __init__(
        self,
        train_txt_file_with_inputs,
        val_txt_file_with_inputs,
        model_output_path,
        learning_rate,
        number_of_epochs,
        images_per_batch,
        batch_size_per_image,
    ) -> None:
        self.train_txt_file_with_inputs = train_txt_file_with_inputs
        self.val_txt_file_with_inputs = val_txt_file_with_inputs
        self.model_output_path = model_output_path
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.images_per_batch = images_per_batch
        self.batch_size_per_image = batch_size_per_image

    def train(self):
        # Setup loger
        setup_logger()

        for d in [self.train_txt_file_with_inputs, self.val_txt_file_with_inputs]:
            if not (f"mud_{d.stem}" in MetadataCatalog.data.keys()):
                DatasetCatalog.register(f"mud_{d.stem}", lambda d=d: get_dict(d))
                MetadataCatalog.get(f"mud_{d.stem}").set(
                    thing_classes=["clear", "transparent", "semi-transparent", "opaque"],
                    stuff_classes=["clear", "transparent", "semi-transparent", "opaque"],
                    ignore_label=False,
                )
        set_seed()
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("Misc/semantic_R_50_FPN_1x.yaml"))
        cfg.DATASETS.TRAIN = (f"mud_{self.train_txt_file_with_inputs.stem}",)
        cfg.DATASETS.TEST = (f"mud_{self.val_txt_file_with_inputs.stem}",)
        if self.number_of_epochs > 100:
            cfg.TEST.EVAL_PERIOD = 100
        cfg.DATALOADER.NUM_WORKERS = 2
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/semantic_R_50_FPN_1x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = self.images_per_batch
        cfg.SOLVER.BASE_LR = self.learning_rate  # 0.00025  # pick a good LR 0.00025
        cfg.SOLVER.MAX_ITER = self.number_of_epochs
        cfg.SOLVER.WARMUP_ITERS = int(0.1 * cfg.SOLVER.MAX_ITER)
        # cfg.SOLVER.LR_FACTOR_FUNC = 0.5  # TODO remove
        # cfg.SOLVER.GAMMA = 0.5
        cfg.SOLVER.STEPS = [int(x) for x in np.linspace(0, cfg.SOLVER.MAX_ITER, 11)][1:-1]
        # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.batch_size_per_image  # 128
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 4
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        cfg.OUTPUT_DIR = str(self.model_output_path)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        # trainer = DefaultTrainer(cfg=cfg)
        # trainer = CustomTrainer(cfg=cfg)
        trainer = MyTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train psnet")
    parser.add_argument(
        "--train_txt_file_with_inputs",
        help="Full file path to the txt file with list of training file names",
        default=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/train_all_files.json",
    )
    parser.add_argument(
        "--val_txt_file_with_inputs",
        help="Size of the image. Ration is 1:1, so provided value" "resolution should be as image_size:image_size",
        default=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/train/val_all_files.json",
    )
    parser.add_argument(
        "--model_output_path",
        help="Folder path, where sould be stored models",
        default=r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/model/detectron2",
    )
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Model learning rate")
    parser.add_argument("--number_of_epochs", default=10, help="Number of epochs for model training")
    parser.add_argument("--images_per_batch", type=int, default=8, help="Number of images per batch")
    parser.add_argument("--batch_size_per_image", type=int, default=512, help="Batch size per image on the roi head")
    args = parser.parse_args()
    # Initiate model with parameters
    get_gpu_status()
    Detectron2Trainer(
        train_txt_file_with_inputs=pathlib.Path(args.train_txt_file_with_inputs),
        val_txt_file_with_inputs=pathlib.Path(args.val_txt_file_with_inputs),
        model_output_path=pathlib.Path(args.model_output_path),
        learning_rate=args.learning_rate,
        number_of_epochs=args.number_of_epochs,
        images_per_batch=args.images_per_batch,
        batch_size_per_image=args.batch_size_per_image,
    ).train()
