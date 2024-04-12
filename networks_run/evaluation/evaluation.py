import pathlib
from tqdm import tqdm
import pandas as pd

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_imgs(img, lbl, pred, output_file_path):
    fig, ax = plt.subplots(1, 3, figsize=(18, 10))
    ax[0].imshow(img)
    ax[0].set_title("Original image")
    ax[1].imshow(lbl)
    ax[1].set_title("Annotation")
    ax[2].imshow(pred)
    ax[2].set_title("Prediction")
    plt.savefig(output_file_path)
    plt.close("all")
    # plt.show()


def run_evaluation(
    prediction_path,
    annotation_path,
    img_path,
    output_path,
    cls_names,  # Put them in order of their ids 0 = ["First value"]
    width="",
    height="",
):
    output_stats = {
        cls_name: {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "IoU": 0, "Precision": 0, "Recall": 0} for cls_name in cls_names
    }
    # output_stats["Total_no_of_pixels"] = 0
    prediction_path = pathlib.Path(prediction_path)
    annotation_path = pathlib.Path(annotation_path)
    img_path = pathlib.Path(img_path)
    output_path = pathlib.Path(output_path)
    output_path_stats = output_path / "stats"
    output_path_imgs = output_path / "imgs"
    output_path_stats.mkdir(parents=True, exist_ok=True)
    output_path_imgs.mkdir(parents=True, exist_ok=True)

    lo_annotations = list(annotation_path.glob("*.png"))
    for annotation_path in tqdm(lo_annotations):
        img_ann = Image.open(annotation_path)
        img_pred = Image.open(prediction_path / annotation_path.name)
        img = Image.open(img_path / annotation_path.name)
        if width != "" and height != "":
            img_ann = img_ann.resize((width, height))
            img_pred = img_pred.resize((width, height))
            img = img.resize((width, height))
        img_ann = np.array(img_ann)
        img_pred = np.array(img_pred)
        img = np.array(img)
        save_imgs(
            img=img,
            lbl=img_ann,
            pred=img_pred,
            output_file_path=str(output_path_imgs / (annotation_path.stem + ".png")),
        )
        for class_id in range(len(cls_names)):
            tp = np.sum((img_pred == class_id) & (img_ann == class_id))
            fp = np.sum((img_pred == class_id) & (img_ann != class_id))
            tn = np.sum((img_pred != class_id) & (img_ann != class_id))
            fn = np.sum((img_pred != class_id) & (img_ann == class_id))
            output_stats[cls_names[class_id]]["TP"] += tp
            output_stats[cls_names[class_id]]["FP"] += fp
            output_stats[cls_names[class_id]]["TN"] += tn
            output_stats[cls_names[class_id]]["FN"] += fn
    for class_id in range(len(cls_names)):
        tp = output_stats[cls_names[class_id]]["TP"]
        fp = output_stats[cls_names[class_id]]["FP"]
        tn = output_stats[cls_names[class_id]]["TN"]
        fn = output_stats[cls_names[class_id]]["FN"]

        output_stats[cls_names[class_id]]["IoU"] += tp / (tp + fp + fn)
        output_stats[cls_names[class_id]]["Precision"] += tp / (tp + fp)
        output_stats[cls_names[class_id]]["Recall"] += tp / (tp + fn)
    # output_stats["Total_no_of_pixels"] += len(img_pred.reshape(-1))
    df = pd.DataFrame.from_dict(output_stats)
    df.to_csv(output_path_stats / "absolute_stats.csv")
    pd.concat(
        [
            df.loc[["TP", "TN", "FP", "FN"]] / df.loc[["TP", "TN", "FP", "FN"]].sum(axis=0),
            df.loc[["IoU", "Precision", "Recall"]],
        ]
    ).to_csv(output_path_stats / "relative_stats.csv")


if __name__ == "__main__":
    run_evaluation(
        prediction_path=r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/resnet50_unet_train_enc_correct_clear_strict_files/predictions",
        annotation_path=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/gtLabels",
        img_path=r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling/test/rgbImages",
        output_path=r"/home/fberanek/Desktop/learning/my_articles/outputs/keras/resnet50_unet_train_enc_correct_clear_strict_files/evaluations",
        width=512,
        height=512,
        cls_names=["Clear", "Transparent", "Semi-Transparent", "Opaque"],
    )
