import os
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
    tf_limit_gpu_memory(tf, 4500)

    # Load Mask-RCNN config
    from common.config import CONFIG

    CONFIG.update(coco.COCO_CONFIG)

    # Set only 5 COCO classes
    # [512, 576, 640, 704, 768, 832, 896, 960, 1024]
    CONFIG.update({'class_dict': {'background': 0,
                                  'person': 1,
                                  'bicycle': 2,
                                  'car': 3,
                                  'motorcycle': 4,
                                  },
                   'num_classes': 5,
                   'meta_shape': 1 + 3 + 3 + 4 + 1 + 5,  # 4 COCO classes + 1 background class
                   'image_shape': (1024, 1024, 3),
                   'image_resize_mode': 'square',
                   'img_size': 1024,
                   'image_min_dim': 800,
                   'image_min_scale': 0,
                   'image_max_dim': 1024,

                   'backbone': 'mobilenet',
                   'epochs': 50,
                   'batch_size': 1,
                   'images_per_gpu': 1,
                   'train_bn': False,

                   }
                  )

    # Init training and validation datasets
    base_dir = r''
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    # Initialize training and validation datasets

    train_dataset = coco.CocoDataset(dataset_dir=train_dir,
                                     subset='train',
                                     class_ids=[1, 2, 3, 4],
                                     year=2017,
                                     auto_download=False,

                                     # SegmentationDataset necessary parent attributes
                                     augmentation=aug.get_training_augmentation(
                                         image_size=CONFIG['img_size'],
                                         normalize=CONFIG['normalization']
                                     ),
                                     **CONFIG
                                     )

    val_dataset = coco.CocoDataset(dataset_dir=val_dir,
                                   subset='val',
                                   class_ids=[1, 2, 3, 4],
                                   year=2017,
                                   auto_download=False,

                                   # SegmentationDataset necessary parent attributes
                                   augmentation=aug.get_validation_augmentation(
                                       image_size=CONFIG['img_size'],
                                       normalize=CONFIG['normalization']
                                   ),
                                   **CONFIG
                                   )

    # Use only 1000 random images for train and 100 random images for validation
    train_imgs = 1000
    val_imgs = 100
    random.shuffle(train_dataset.images_names)
    random.shuffle(val_dataset.images_names)
    train_dataset.images_names = train_dataset.images_names[:train_imgs]
    val_dataset.images_names = val_dataset.images_names[:val_imgs]

    # Init Mask-RCNN model
    model = mask_rcnn_functional(config=CONFIG)

    # Train
    train_model(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=CONFIG,
                weights_path=None)
