from keras_networks.train import (
    psp_101_train,
    resnet50_pspnet_train,
    resnet50_segnet_train,
    resnet50_unet_train,
)
from keras_networks.predict import (
    psp_101,
    resnet50_pspnet,
    resnet50_segnet,
    resnet50_unet,
)
from detectron.train.detectron2_train import Detectron2Trainer
from detectron.predict import detectron2_pred
from evaluation import evaluation
import pathlib


train_data_root = pathlib.Path(
    r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling"
)
models_output = pathlib.Path(r"/home/fberanek/Desktop/learning/my_articles/outputs")
val_coeficient = 0.1
learning_rate = 0.0001
number_of_epochs = 1000
batch_size = 2
keras_models = []
segnet = False
unet = True
psp_resnet = False
psp = False
detetron2 = False

list_of_files = ["all_files", 
                "correct_files",
                "correct_clear_files",
                "correct_clear_strict_files"]

for text_file in list_of_files[2:]:
    detectron2_txt_train_file = train_data_root / "train" / f"train_{text_file}.json"
    detectron2_txt_val_file = train_data_root / "train" / f"val_{text_file}.json"
    if detetron2:
        Detectron2Trainer(
            root = train_data_root / "train",
            model_output_path = str(models_output / f"detectron2_{text_file}" / "model"),
            dataset = [f"train_{text_file}", f"val_{text_file}"],
            max_iteration= int(number_of_epochs * 5),
            learning_rate = learning_rate/10
            ).train()

for text_file in list_of_files:
    keras_txt_train_file = train_data_root / "train" / f"train_{text_file}.txt"
    keras_txt_val_file = train_data_root / "train" / f"val_{text_file}.txt"
    detectron2_txt_train_file = train_data_root / "train" / f"train_{text_file}.json"
    detectron2_txt_val_file = train_data_root / "train" / f"val_{text_file}.json"
    if unet:
        resnet50_unet_train.PredefinedModel(
        train_txt_file_with_inputs = keras_txt_train_file,
        val_txt_file_with_inputs = keras_txt_val_file,
        dataset_root = train_data_root, 
        model_output_path = models_output 
                / "keras"
                / f"resnet50_unet_train_enc_{text_file}"
                / "model",
        learning_rate = learning_rate,
        number_of_epochs = number_of_epochs,
        width = 512,
        height = 512,
        batch_size = batch_size,
        encoder_level = 3).train_model()
    for encoder_level in [2,3,4]: # Head must be changed for  3, 5, 8, 12, 20]
        if segnet:
            # Run Segnet
            resnet50_segnet_train.PredefinedModel(
            train_txt_file_with_inputs = keras_txt_train_file,
            val_txt_file_with_inputs = keras_txt_val_file,
            dataset_root = train_data_root, 
            model_output_path = models_output 
                    / "keras"
                    / f"resnet50_segnet_train_enc_lvl_{encoder_level}_{text_file}"
                    / "model",
            learning_rate = learning_rate,
            number_of_epochs = number_of_epochs,
            width = 512,
            height = 512,
            batch_size = batch_size,
            encoder_level = encoder_level).train_model()
    if psp_resnet:
        # Run PSP50
        resnet50_pspnet_train.PredefinedModel(
            train_txt_file_with_inputs = keras_txt_train_file,
            val_txt_file_with_inputs = keras_txt_val_file,
            dataset_root = train_data_root, 
            model_output_path = models_output 
                    / "keras"
                    / f"resnet50_pspnet_train_{text_file}"
                    / "model",
            learning_rate = learning_rate,
            number_of_epochs = number_of_epochs,
            width=576,
            height=384,
            batch_size=batch_size,
        ).train_model()
    if psp:
        # Run PSP101
        psp_101_train.PredefinedModel(
            train_txt_file_with_inputs = keras_txt_train_file,
            val_txt_file_with_inputs = keras_txt_val_file,
            dataset_root = train_data_root, 
            model_output_path = models_output 
                    / "keras"
                    / f"resnet50_segnet_train_enc_{text_file}"
                    / "model",
            learning_rate = learning_rate,
            number_of_epochs = number_of_epochs,
            width=473,
            height=473,
            batch_size=batch_size
        ).train_model()

'''

resnet50_segnet.PredictResnetUnet(
    images_folder_path=train_data_root / "test" / "rgbImages",
    model_path=models_output
    / "keras"
    / f"resnet50_segnet_train_enc_lvl_{encoder_level}_{file_txt}"
    / "model"
    / "checkpoint.model.h5",
    output_path=models_output
    / "keras"
    / f"resnet50_segnet_train_enc_lvl_{encoder_level}_{file_txt}"
    / "predictions",
    width = 512,
    height = 512,
    encoder_level = encoder_level
).predict()
resnet50_unet.PredictResnetUnet(
    train_data_root / "test" / "rgbImages",
    models_output
    / "keras"
    / f"resnet50_unet_train_enc_lvl_{encoder_level}_{file_txt}"
    / "model"
    / "checkpoint.model.h5",
    models_output
    / "keras"
    / f"resnet50_unet_train_enc_lvl_{encoder_level}_{file_txt}"
    / "predictions",
    512,
).predict()
psp_101.PredictResnetUnet(
    train_data_root / "test" / "rgbImages",
    models_output
    / "keras"
    / f"psp_101_train_{file_txt}"
    / "model"
    / "checkpoint.model.h5",
    models_output / "keras" / f"psp_101_train_{file_txt}" / "predictions",
    473,
).predict()
resnet50_pspnet.PredictResnetUnet(
    train_data_root / "test" / "rgbImages",
    models_output
    / "keras"
    / f"resnet50_pspnet_train_{file_txt}"
    / "model"
    / "checkpoint.model.h5",
    models_output
    / "keras"
    / f"resnet50_pspnet_train_{file_txt}"
    / "predictions",
    473,
).predict()
detectron2_pred.predict(
    train_data_root / "test" / "rgbImages",
    models_output
    / f"detectron2_{text_file}"
    / "model"
    / "model_train"
    / "model_final.pth",
    models_output / f"detectron2_{text_file}" / "predictions",
)
'''