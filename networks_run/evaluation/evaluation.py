import pathlib
from tqdm import tqdm
import pandas as pd

import numpy as np
from PIL import Image


def run_evaluation(prediction_path,
                   annotation_path,
                   output_path,
                   cls_names,  # Put them in order of their ids 0 = ["First value"]
                   ):
    output_stats = {cls_name: {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    } for cls_name in cls_names}
    # output_stats["Total_no_of_pixels"] = 0
    prediction_path = pathlib.Path(prediction_path)
    annotation_path = pathlib.Path(annotation_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    lo_annotations = annotation_path.glob("*.png")
    for annotation_path in tqdm(lo_annotations):
        img_ann = np.array(Image.open(annotation_path))
        img_pred = np.array(Image.open(prediction_path/annotation_path.name))
        for class_id in range(len(cls_names)):
            tp = np.sum((img_pred == class_id) & (img_ann == class_id))
            fp = np.sum((img_pred == class_id) & (img_ann != class_id))
            tn = np.sum((img_pred != class_id) & (img_ann != class_id))
            fn = np.sum((img_pred != class_id) & (img_ann == class_id))
            output_stats[cls_names[class_id]]["TP"] += tp
            output_stats[cls_names[class_id]]["FP"] += fp
            output_stats[cls_names[class_id]]["TN"] += tn
            output_stats[cls_names[class_id]]["FN"] += fn
            # output_stats["Total_no_of_pixels"] += len(img_pred.reshape(-1))
    pd.DataFrame.from_dict(output_stats).to_csv(output_path/"absolute_stats.csv")
    df = pd.DataFrame.from_dict(output_stats)
    (df/df.sum(axis=0)).to_csv(output_path/"relative_stats.csv")


run_evaluation(prediction_path=r"/home/fberanek/Desktop/learning/my_articles/outputs/detectron2/predictions",
               annotation_path=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/gtLabels",
               output_path=r"/home/fberanek/Desktop/learning/my_articles/outputs/detectron2/evaluation",
               cls_names=["Clear", "Transparent", "Semi-Transparent", "Opaque"])
