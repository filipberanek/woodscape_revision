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
    
    input_path_test = input_path / "test"
    records = open(input_path_test/"test_all_files.txt").readlines()
    list_of_labels = [f"{input_path_test}{row.split(',')[1].strip()}" for row in records]
    list_of_iamges = [f"{input_path_test}{row.split(',')[0].strip()}" for row in records]
    create_coco_annotation(list_of_iamges, list_of_labels, input_path_test, "test_all_files")
    print(f"len of test records {len(list_of_iamges)}")

    input_path_train = input_path / "train"
    records = open(input_path_train/"train_all_files.txt").readlines()
    list_of_labels = [f"{input_path_train}{row.split(',')[1].strip()}" for row in records]
    list_of_iamges = [f"{input_path_train}{row.split(',')[0].strip()}" for row in records]
    create_coco_annotation(list_of_iamges, list_of_labels, input_path_train, "train_all_files")
    print(f"len of train_all_files records {len(list_of_iamges)}")
    records = open(input_path_train/"val_all_files.txt").readlines()
    list_of_labels = [f"{input_path_train}{row.split(',')[1].strip()}" for row in records]
    list_of_iamges = [f"{input_path_train}{row.split(',')[0].strip()}" for row in records]
    create_coco_annotation(list_of_iamges, list_of_labels, input_path_train, "val_all_files")
    print(f"len of val_all_files records {len(list_of_iamges)}")

    records = open(input_path_train/"train_correct_files.txt").readlines()
    list_of_labels = [f"{input_path_train}{row.split(',')[1].strip()}" for row in records]
    list_of_iamges = [f"{input_path_train}{row.split(',')[0].strip()}" for row in records]
    create_coco_annotation(list_of_iamges, list_of_labels, input_path_train, "train_correct_files")
    print(f"len of train_correct records {len(list_of_iamges)}")
    records = open(input_path_train/"val_correct_files.txt").readlines()
    list_of_labels = [f"{input_path_train}{row.split(',')[1].strip()}" for row in records]
    list_of_iamges = [f"{input_path_train}{row.split(',')[0].strip()}" for row in records]
    create_coco_annotation(list_of_iamges, list_of_labels, input_path_train, "val_correct_files")
    print(f"len of val_correct records {len(list_of_iamges)}")

    records = open(input_path_train/"train_correct_clear_files.txt").readlines()
    list_of_labels = [f"{input_path_train}{row.split(',')[1].strip()}" for row in records]
    list_of_iamges = [f"{input_path_train}{row.split(',')[0].strip()}" for row in records]
    create_coco_annotation(list_of_iamges, list_of_labels, input_path_train, "train_correct_clear_files")
    print(f"len of train_correct_clear records {len(list_of_iamges)}")
    records = open(input_path_train/"val_correct_clear_files.txt").readlines()
    list_of_labels = [f"{input_path_train}{row.split(',')[1].strip()}" for row in records]
    list_of_iamges = [f"{input_path_train}{row.split(',')[0].strip()}" for row in records]
    create_coco_annotation(list_of_iamges, list_of_labels, input_path_train, "val_correct_clear_files")
    print(f"len of val_correct_clear records {len(list_of_iamges)}")

    records = open(input_path_train/"train_correct_clear_strict_files.txt").readlines()
    list_of_labels = [f"{input_path_train}{row.split(',')[1].strip()}" for row in records]
    list_of_iamges = [f"{input_path_train}{row.split(',')[0].strip()}" for row in records]
    create_coco_annotation(list_of_iamges, list_of_labels, input_path_train, "train_correct_clear_strict_files")
    print(f"len of train_correct_clear_strict records {len(list_of_iamges)}")
    records = open(input_path_train/"val_correct_clear_strict_files.txt").readlines()
    list_of_labels = [f"{input_path_train}{row.split(',')[1].strip()}" for row in records]
    list_of_iamges = [f"{input_path_train}{row.split(',')[0].strip()}" for row in records]
    create_coco_annotation(list_of_iamges, list_of_labels, input_path_train, "val_correct_clear_strict_files")
    print(f"len of val_correct_clear_strict records {len(list_of_iamges)}")
    # Run test
    # input_path_test = input_path / "test"
    # records = open(input_path_train/"correct_clear_strict.txt").readlines()
    # list_of_labels = [f"{input_path_test}{row.split(',')[1].strip()}" for row in records]
    # list_of_iamges = [f"{input_path_test}{row.split(',')[0].strip()}" for row in records]
    # create_coco_annotation(list_of_iamges, list_of_labels, input_path_train, "train_correct_clear_strict")
