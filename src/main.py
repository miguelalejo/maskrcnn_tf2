import os

import tensorflow as tf

from common.config import CONFIG
from common.utils import tf_limit_gpu_memory
from model import mask_rcnn_functional
from preprocess import augmentation as aug
from preprocess import preprocess
from training import train_model

if __name__ == '__main__':
    # Model training

    tf_limit_gpu_memory(tf, 4500)
    base_dir = os.getcwd().replace('src', 'licence_segmentation_dataset')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    model = mask_rcnn_functional(config=CONFIG)

    train_dataset = preprocess.SegmentationDataset(images_dir=train_dir,
                                                   class_key='object',
                                                   classes_dict=CONFIG['class_dict'],
                                                   augmentation=aug.get_training_augmentation(
                                                       image_size=CONFIG['img_size'],
                                                       normalize=CONFIG['normalization']
                                                   ),
                                                   **CONFIG
                                                   )

    val_dataset = preprocess.SegmentationDataset(images_dir=val_dir,
                                                 class_key='object',
                                                 classes_dict=CONFIG['class_dict'],
                                                 augmentation=aug.get_validation_augmentation(
                                                     image_size=CONFIG['img_size'],
                                                     normalize=CONFIG['normalization']
                                                 ),
                                                 **CONFIG
                                                 )

    train_model(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=CONFIG,
                weights_path=None)
