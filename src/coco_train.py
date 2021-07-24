import multiprocessing as mp
import random

import tensorflow as tf

from common.utils import tf_limit_gpu_memory
from model import mask_rcnn_functional
from preprocess import augmentation as aug
from samples.coco import coco
from training import train_model

if __name__ == '__main__':
    # Init random seed
    random.seed(42)

    # Limit GPU memory for tensorflow container
    tf_limit_gpu_memory(tf, 7500)

    # Load Mask-RCNN config
    from common.config import CONFIG

    CONFIG.update(coco.COCO_CONFIG)
    CONFIG.update({'image_shape': (1024, 1024, 3),
                   'image_resize_mode': 'square',
                   'img_size': 1024,
                   'image_max_dim': 1024,
                   'backbone': 'mobilenet',
                   'epochs': 150,
                   'batch_size': 2,
                   'images_per_gpu': 2,
                   'train_bn': True,
                   'use_multiprocessing': True,
                   'workers': mp.cpu_count()

                   }
                  )

    # Set folder for coco dataset
    base_dir = r''

    # Initialize training and validation datasets
    train_dataset = coco.CocoDataset(dataset_dir=base_dir,
                                     subset='train',
                                     year=2017,
                                     auto_download=True,
                                     # SegmentationDataset necessary parent attributes
                                     augmentation=aug.get_training_augmentation(
                                         image_size=CONFIG['img_size'],
                                         normalize=CONFIG['normalization']
                                     ),
                                     **CONFIG
                                     )

    val_dataset = coco.CocoDataset(dataset_dir=base_dir,
                                   subset='val',
                                   year=2017,
                                   auto_download=True,
                                   # SegmentationDataset necessary parent attributes
                                   augmentation=aug.get_validation_augmentation(
                                       image_size=CONFIG['img_size'],
                                       normalize=CONFIG['normalization']
                                   ),
                                   **CONFIG
                                   )

    # Init Mask-RCNN model
    model = mask_rcnn_functional(config=CONFIG)

    # Train
    train_model(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=CONFIG,
                weights_path=None
                )
