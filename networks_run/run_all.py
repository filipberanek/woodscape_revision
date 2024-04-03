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
from detectron.train import detectron2_train
from detectron.predict import detectron2_pred
from evaluation import evaluation
import pathlib


train_data_root = pathlib.Path(
    r"/home/fberanek/Desktop/datasets/segmentation/semantic/new_soiling"
)
models_output = pathlib.Path(r"/home/fberanek/Desktop/learning/my_articles/outputs")
val_coeficient = 0.1
learning_rate = 0.0001
number_of_epochs = 1
batch_size = 2
keras_models = []

for text_file in ["correct.txt", "correct_clear.txt", "correct_clear_strict.txt"]:
    # Run PSP101
    psp_101_train.PredefinedModel(
        train_data_root / "train" / text_file,
        train_data_root,
        models_output / "keras" / f"psp_101_train_{text_file}" / "model",
        val_coeficient,
        learning_rate,
        number_of_epochs,
        473,
        batch_size,
    ).train_model()
    psp_101.PredictResnetUnet(
        train_data_root / "test" / "rgbImages",
        models_output
        / "keras"
        / f"psp_101_train_{text_file}"
        / "model"
        / "checkpoint.model.h5",
        models_output / "keras" / f"psp_101_train_{text_file}" / "predictions",
        473,
    ).predict()
    # Run PSP50
    resnet50_pspnet_train.PredefinedModel(
        train_data_root / "train" / text_file,
        train_data_root,
        models_output / "keras" / f"resnet50_pspnet_train_{text_file}" / "model",
        val_coeficient,
        learning_rate,
        number_of_epochs,
        473,
        batch_size,
    ).train_model()
    resnet50_pspnet.PredictResnetUnet(
        train_data_root / "test" / "rgbImages",
        models_output
        / "keras"
        / f"resnet50_pspnet_train_{text_file}"
        / "model"
        / "checkpoint.model.h5",
        models_output / "keras" / f"resnet50_pspnet_train_{text_file}" / "predictions",
        473,
    ).predict()
    for encoder_level in [2, 3, 5, 8, 12, 20]:
        # Run Segnet
        resnet50_segnet_train.PredefinedModel(
            train_data_root / "train" / text_file,
            train_data_root,
            models_output
            / "keras"
            / f"resnet50_segnet_train_enc_lvl_{encoder_level}_{text_file}"
            / "model",
            val_coeficient,
            learning_rate,
            number_of_epochs,
            512,
            batch_size,
            encoder_level,
        ).train_model()
        resnet50_segnet.PredictResnetUnet(
            train_data_root / "test" / "rgbImages",
            models_output
            / "keras"
            / f"resnet50_segnet_train_enc_lvl_{encoder_level}_{text_file}"
            / "model"
            / "checkpoint.model.h5",
            models_output
            / "keras"
            / f"resnet50_segnet_train_enc_lvl_{encoder_level}_{text_file}"
            / "predictions",
            512,
        ).predict()
        # Run Unet
        resnet50_unet_train.PredefinedModel(
            train_data_root / "train" / text_file,
            train_data_root,
            models_output
            / "keras"
            / f"resnet50_unet_train_enc_lvl_{encoder_level}_{text_file}",
            val_coeficient,
            learning_rate,
            number_of_epochs,
            512,
            batch_size,
            encoder_level,
        ).train_model()
        resnet50_unet.PredictResnetUnet(
            train_data_root / "test" / "rgbImages",
            models_output
            / "keras"
            / f"resnet50_unet_train_enc_lvl_{encoder_level}_{text_file}"
            / "model"
            / "checkpoint.model.h5",
            models_output
            / "keras"
            / f"resnet50_unet_train_enc_lvl_{encoder_level}_{text_file}"
            / "predictions",
            512,
        ).predict()
    detectron2_train.train(
        train_data_root / "train",
        models_output / f"detectron2_{text_file}" / "model",
        text_file,
        int(number_of_epochs * 20),
    )
    detectron2_pred.predict(
        train_data_root / "test" / "rgbImages",
        models_output
        / f"detectron2_{text_file}"
        / "model"
        / "model_train"
        / "model_final.pth",
        models_output / f"detectron2_{text_file}" / "predictions",
    )
