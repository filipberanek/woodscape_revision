import pathlib
from PIL import Image
import json
import argparse


def create_coco_annotation(lo_images, lo_labels, output_folder_path, keyword):

    output_dict = []
    for img_id, (image_path, label_path) in enumerate(zip(lo_images, lo_labels)):
        img = Image.open(image_path)
        single_records = {
            "image_id": img_id,
            "file_name": str(image_path),
            "width": img.size[0],
            "height": img.size[1],
            "sem_seg_file_name": str(label_path)
        }
        output_dict.append(single_records)
    with open(output_folder_path/f"{keyword}.json", "w") as outfile:
        json.dump(output_dict, outfile)


if __name__ == "__main__":
    input_path = pathlib.Path(r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling")
    # Run test
    input_path_test = input_path / "test"
    l_o_images = list((input_path_test / "rgbImages").rglob("*.png"))
    l_o_labels = list((input_path_test / "gtLabels").rglob("*.png"))
    create_coco_annotation(l_o_images, l_o_labels, input_path_test, "test")
    # Run train
    input_path_train = input_path / "train"
    split = -497
    l_o_images = list((input_path_test / "rgbImages").rglob("*.png"))[:split]
    l_o_labels = list((input_path_test / "gtLabels").rglob("*.png"))[:split]
    create_coco_annotation(l_o_images, l_o_labels, input_path_train, "train")
    # Run val
    l_o_images = list((input_path_test / "rgbImages").rglob("*.png"))[split:]
    l_o_labels = list((input_path_test / "gtLabels").rglob("*.png"))[split:]
    create_coco_annotation(l_o_images, l_o_labels, input_path_train, "val")
