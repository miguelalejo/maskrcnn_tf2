import argparse
import os
import random
import sys

import tensorflow as tf

from common.config import CONFIG
from common.utils import tf_limit_gpu_memory
from model import mask_rcnn_functional
from preprocess import augmentation as aug
from preprocess import preprocess as prep
from samples.coco import coco
from training import train_model


def coco_minitrain(pargs: argparse.Namespace) -> None:
    random.seed(pargs.rseed)
    tf_limit_gpu_memory(tf, pargs.gpu_memory)

    CONFIG.update(coco.COCO_CONFIG)
    # Set only 5 COCO classes
    CONFIG.update({'class_dict': {'background': 0,
                                  'person': 1,
                                  'bicycle': 2,
                                  'car': 3,
                                  'motorcycle': 4,
                                  },
                   'num_classes': 5,
                   'meta_shape': 1 + 3 + 3 + 4 + 1 + 5,  # 4 COCO classes + 1 background class
                   'image_shape': (pargs.image_size, pargs.image_size, 3),
                   'image_resize_mode': 'square',
                   'img_size': pargs.image_size,
                   'image_min_dim': pargs.image_size,
                   'image_min_scale': 0,
                   'image_max_dim': pargs.image_size,

                   'backbone': pargs.backbone,
                   'epochs': pargs.epochs,
                   'batch_size': pargs.batch_size,
                   'images_per_gpu': pargs.batch_size,
                   'train_bn': pargs.train_bn,

                   'frozen_backbone': pargs.frozen_backbone,

                   'callback': {
                       # TensorBoard callback
                       'checkpoints_dir': pargs.checkpoints_path,
                       # ReduceLROnPlateau callback
                       'reduce_lr_on_plateau': 0.98,
                       'reduce_lr_on_plateau_patience': 10,
                       # ModelCheckpoint callback
                       'save_weights_only': True,
                       'save_best_only': True,
                       'histogram_freq': 0,
                       'profile_batch': '1,2',
                   },

                   }
                  )

    # Init training and validation datasets
    train_dir = os.path.join(pargs.dataset_path, 'train')
    val_dir = os.path.join(pargs.dataset_path, 'val')

    # Initialize training and validation datasets
    train_dataset = coco.CocoDataset(dataset_dir=train_dir,
                                     subset='train',
                                     class_ids=(1, 2, 3, 4),
                                     year=2017,
                                     auto_download=False,
                                     preprocess_transform=prep.get_input_preprocess(
                                         normalize=CONFIG['normalization']
                                     ),
                                     augmentation=aug.get_training_augmentation(),
                                     **CONFIG
                                     )

    val_dataset = coco.CocoDataset(dataset_dir=val_dir,
                                   subset='val',
                                   class_ids=(1, 2, 3, 4),
                                   year=2017,
                                   auto_download=False,
                                   preprocess_transform=prep.get_input_preprocess(
                                       normalize=CONFIG['normalization']
                                   ),
                                   **CONFIG
                                   )
    random.shuffle(train_dataset.images_names)
    random.shuffle(val_dataset.images_names)
    train_dataset.images_names = train_dataset.images_names[:pargs.n_train_images]
    val_dataset.images_names = val_dataset.images_names[:pargs.n_val_images]
    # Init Mask-RCNN model
    model = mask_rcnn_functional(config=CONFIG)
    # Train
    train_model(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=CONFIG,
                weights_path=None)


if __name__ == '__main__':
    sys.exit(coco_minitrain(coco.coco_parse_arguments()))
