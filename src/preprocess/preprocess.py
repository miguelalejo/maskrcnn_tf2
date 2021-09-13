import json
import os
import warnings

import cv2
import albumentations as img_album
import numpy as np
import scipy
from common import utils
from tensorflow import cast
from tensorflow.keras.utils import Sequence


class SegmentationDataset:

    def __init__(self, images_dir=None, class_key='object', augmentation=None,
                 preprocess_transform=False, json_annotation_key='_via_img_metadata', **kwargs):
        """
        Dataset class for VGG Image Annotator. Read images, apply augmentation and preprocessing transformations.
        Args:
            images_dir:           (str): path to images folder
            class_key:            (str): class_key may be a key for class name for polygons
            augmentation:         (albumentations.Compose): data transfromation pipeline
            preprocess_transform: (albumentations.Compose):  transformation of an image
            json_annotation_key:  (str): default key to extract annotations from .json.
                                         By default, it is '_via_img_metadata' for VGG Image Annotator
            **kwargs:             additional processing configuration parameters
        """
        super(SegmentationDataset, self).__init__()

        self.kwargs = kwargs
        self.class_key = class_key
        self.json_annotation_key = json_annotation_key

        if images_dir:
            self.images_names = [x for x in os.listdir(images_dir) if '.json' not in x]
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.images_names]

            # Find annotation file and make sure that folder contains only one annotation file
            annot_file = [x for x in os.listdir(images_dir) if '.json' in x]
            assert len(annot_file) == 1
            annot_file = annot_file[0]
            print(f'Found annotation file: {annot_file} in dataset path: {images_dir}')

            if self.json_annotation_key:
                self.annotation_dict = json.load(open(os.path.join(images_dir, annot_file)))[json_annotation_key]
            else:
                self.annotation_dict = json.load(open(os.path.join(images_dir, annot_file)))

            # Make sure that keys in json are equal to images filenames
            # Some versions of VIA may violate this rule
            remapped_annotation_dict = {}
            for k, v in self.annotation_dict.items():
                remapped_annotation_dict.update({v['filename']: v})
            self.annotation_dict.clear()
            self.annotation_dict.update(remapped_annotation_dict)
        else:
            print('None passed to images_dir argument.\n',
                  'This means that the dataset class is a child of SegmentationDataset and its'
                  'behaviour differs from datasets created with VGG Image Annotator.\n',
                  'If it is not true, please, check your class arguments carefully.\n')

        # Get class indexes from class_dict
        self.classes_dict = self.kwargs['class_dict']
        self.class_values = list(self.classes_dict.values())
        self.augmentation = augmentation
        self.preprocess_transform = preprocess_transform

        self.backbone_shapes = utils.compute_backbone_shapes(self.kwargs)
        self.anchors = utils.generate_pyramid_anchors(scales=self.kwargs['rpn_anchor_scales'],
                                                      ratios=self.kwargs['rpn_anchor_ratios'],
                                                      feature_shapes=self.backbone_shapes,
                                                      feature_strides=self.kwargs['backbone_strides'],
                                                      anchor_stride=self.kwargs['rpn_anchor_stride']
                                                      )

    def get_points_from_annotation(self, annotation_key):
        """
         Get polygon points for a segment. [[x1,y1], [x2, y2], ....[]]
        Example:
            {'filename': '250024424orig.jpeg',
             'size': 164044,
             'regions': [{'shape_attributes': {'name': 'polygon',
                'all_points_x': [213, 199, 126, 140],
                'all_points_y': [339, 404, 350, 298]},
               'region_attributes': {'object': 'licence'}},
              {'shape_attributes': {'name': 'polygon',
                'all_points_x': [485, 468, 533, 593, 627, 644, 649, 623, 564, 520],
                'all_points_y': [554, 677, 704, 683, 648, 599, 540, 504, 498, 518]},
               'region_attributes': {'object': 'wheel'}}],
             'file_attributes': {}}
             The key for class names is 'object'
        Args:
            annotation_key: key to get info about polygons to make masks

        Returns: polygon_data_list, class_id_list
        """

        polygon_data_list = []
        class_id_list = []
        _region_list = self.annotation_dict[annotation_key]['regions']
        # If there is more than one object described as polygons, find each class id for each polygon
        # If there is no information about classed in 'region_attributes', add class 1 as binary

        for region in _region_list:
            if 'all_points_x' not in region['shape_attributes'].keys():
                print(f'\n[SegmentationDataset] Skipping incorrect observation:\n',
                      f"""annotation_key: {annotation_key}\n_region_list: {region['shape_attributes']}\n""")
                continue
            polygon_points = [[x, y] for x, y in zip(region['shape_attributes']['all_points_x'],
                                                     region['shape_attributes']['all_points_y']
                                                     )
                              ]
            polygon_data_list.append(np.array([polygon_points]))

            # If there is no any keyfields for classes, mark everything as class 1
            if len(region['region_attributes'].keys()) == 0:
                class_id_list.append(1)
            else:
                # In VGG Image Annotator there is an option to add attributes for polygons.
                # We can write class_name to the specified attribute of a polygon
                # For example, by default, attribute name which contains class name is 'object'
                class_name = region['region_attributes'][self.class_key]
                if len(class_name) == 0:
                    raise ValueError(f'Class name is empty. Full annotation: {_region_list}')
                class_id_list.append(self.classes_dict[class_name])

        return polygon_data_list, class_id_list

    def create_mask(self, image, idx):
        """
        Create mask image from VGG Image Annotator metadata
        Args:
            image: original image. numpy array,
            idx:   annotation key to get polygon info about mask

        Returns:  masks_array: A bool array of shape [height, width, instance count] with one mask per instance.
                  class_ids_array: class ids array for each mask

        """
        annotation_key = self.images_names[idx]  # Get image name as annotation key in annotation_dict
        points_list, class_id_list = self.get_points_from_annotation(annotation_key)
        mask_template = np.zeros(image.shape[:2])  # Create mask template with grayscale shape=(width, height)
        instance_masks_list = []
        # Generate one mask per instance
        for points, class_id in zip(points_list, class_id_list):
            instance_masks_list.append(cv2.fillPoly(mask_template, points, (class_id)))
        masks_array = np.stack(instance_masks_list, axis=2).astype(np.bool)  # (w, h, array index)
        class_ids_array = np.array(class_id_list, dtype=np.int32)
        return masks_array, class_ids_array

    def load_image(self, image_id):
        return cv2.imread(self.images_fps[image_id])

    def resize_mask(self, mask, scale, padding, crop=None):
        """Resizes a mask using the given scale and padding.
        Typically, you get the scale and padding from resize_image() to
        ensure both, the image and the mask, are resized consistently.

        scale: mask scaling factor
        padding: Padding to add to the mask in the form
                [(top, bottom), (left, right), (0, 0)]
        """
        # Suppress warning from scipy 0.13.0, the output shape of zoom() is
        # calculated with round() instead of int()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        if crop is not None:
            y, x, h, w = crop
            mask = mask[y:y + h, x:x + w]
        else:
            mask = np.pad(mask, padding, mode='constant', constant_values=0)
        return mask

    def __getitem__(self, id):
        """
        Generate item
        Args:
            id: index of the image to read
        Returns: image, mask, bbox, image_meta, class_ids

        """
        image = self.load_image(id)  # Read image
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Image to RGB color space
        original_image_shape = original_image.shape

        if self.preprocess_transform:
            image = self.preprocess_transform(image=original_image)['image']

        original_masks_array, class_ids_array = self.create_mask(image, id)  # Create image masks from annotation

        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=self.kwargs['image_min_dim'],
            min_scale=self.kwargs['image_min_scale'],
            max_dim=self.kwargs['image_max_dim'],
            mode=self.kwargs['image_resize_mode'])

        masks_array = self.resize_mask(original_masks_array, scale, padding, crop)

        # Apply augmentation
        _image_shape = image.shape

        if self.augmentation:
            masks_list = [masks_array[:, :, i].astype('float') for i in range(masks_array.shape[2])]
            transformed = self.augmentation(image=image, masks=masks_list)
            proc_image, proc_masks = transformed['image'], transformed['masks']

            assert proc_image.shape == _image_shape
            proc_masks = np.stack(proc_masks, axis=2)
        else:
            proc_image = image
            proc_masks = masks_array

        # Note that some boxes might be all zeros if the corresponding mask got cropped out.
        # and here is to filter them out
        _idx = np.sum(proc_masks, axis=(0, 1)) > 0
        proc_masks = proc_masks[:, :, _idx]
        proc_class_ids = class_ids_array[_idx]

        _orig_idx = np.sum(original_masks_array, axis=(0, 1)) > 0
        original_masks_array = original_masks_array[:, :, _orig_idx]
        original_class_ids = class_ids_array[_orig_idx]

        # Compute bboxes
        bboxes = utils.extract_bboxes(proc_masks)
        original_bboxes = utils.extract_bboxes(original_masks_array)

        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        active_class_ids = np.zeros([len(self.classes_dict.keys())], dtype=np.int32)

        # 1 for classes that are in the dataset of the image
        # 0 for classes that are not in the dataset.
        # The position of ones and zeros means the class index.
        source_class_ids = list(
            self.classes_dict.values())  # self.classes_dict['licence'] or list(self.classes_dict.values())
        active_class_ids[source_class_ids] = 1

        # Resize masks to smaller size to reduce memory usage
        if self.kwargs['use_mini_masks']:
            proc_masks = utils.minimize_mask(bboxes, proc_masks, self.kwargs['mini_mask_shape'])

        # Image meta data
        image_meta = utils.compose_image_meta(id, original_image_shape, window, scale, active_class_ids, self.kwargs)

        return proc_image, proc_masks, proc_class_ids, bboxes, image_meta, \
               original_image, original_masks_array, original_class_ids, original_bboxes

    def __len__(self):
        return len(self.images_names)


class DataLoader(Sequence):
    """Load data from dataset and form batches

    Args:
        dataset:    Instance of Dataset class for image loading and preprocessing.
        detection_targets: If True, generate detection targets (class IDs, bbox
                           deltas, and masks). Typically for debugging or visualizations because
                           in training detection targets are generated by DetectionTargetLayer.
        shuffle:    Boolean, if `True` shuffle image indexes each epoch.
        seed:       Seed for pseudo-random generator
        name:       DataLoader name
        cast_output: Cast output to tensorflow.float32
        return_original: Return original images in batch
    """

    def __init__(self, dataset, detection_targets=False, shuffle=True, seed=42, name='dataloader',
                 cast_output=True, return_original=False, **kwargs):

        self.seed = seed
        np.random.seed(self.seed)

        self.dataset = dataset
        self.random_rois = kwargs['random_rois']
        self.detection_targets = detection_targets
        self.indexes = np.arange(len(self.dataset))
        self.anchors = self.dataset.anchors
        self.backbone_shapes = self.dataset.backbone_shapes
        self.shuffle = shuffle
        self.cast_output = cast_output
        self.kwargs = kwargs
        self.batch_size = self.kwargs['batch_size']
        self.return_original = return_original

        self.on_epoch_end()

        self.name = name
        self.steps_per_epoch = self.__len__() // self.batch_size
        print(f'{self.name} DataLoader. Steps per epoch: {self.steps_per_epoch}')

    def generate_batch(self, index):
        """
        Args:
            index: int to get an image

        Returns: python list
                   'batch_images':       tf.random.uniform(shape=(batch, 512, 512, 3),   dtype=tf.float32),
                   'batch_images_meta':  tf.random.uniform(shape=(batch, 14),            dtype=tf.float32),
                   'batch_rpn_match':    tf.random.uniform(shape=(batch, 65472, 1),      dtype=tf.float32),
                   'batch_rpn_bbox':     tf.random.uniform(shape=(batch, 256, 4),        dtype=tf.float32),
                   'batch_gt_class_ids': tf.random.uniform(shape=(batch, 100),           dtype=tf.float32),
                   'batch_gt_boxes':     tf.random.uniform(shape=(batch, 100, 4),        dtype=tf.float32),
                   'batch_gt_masks':     tf.random.uniform(shape=(batch, 512, 512, 100), dtype=tf.float32),

        """
        # Set batch size counter
        gen_batch = 0
        while gen_batch < self.batch_size:

            image, gt_masks, gt_class_ids, gt_boxes, image_meta, \
            original_image, original_masks_array, original_class_ids, original_bboxes = self.dataset[index]
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                index = min(index + 1, len(self.indexes) - 1)
                continue

            # RPN Targets
            rpn_match, rpn_bbox = utils.build_rpn_targets(
                anchors=self.anchors,
                gt_class_ids=gt_class_ids,
                gt_boxes=gt_boxes,
                rpn_train_anchors_per_image=self.kwargs['rpn_train_anchors_per_image'],
                rpn_bbox_std=self.kwargs['rpn_bbox_std_dev']
            )

            # Mask R-CNN Targets
            if self.random_rois:
                rpn_rois = utils.generate_random_rois(image.shape, self.random_rois, gt_boxes)
                if self.detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = utils.build_detection_targets(
                        rpn_rois=rpn_rois, gt_class_ids=gt_class_ids, gt_boxes=gt_boxes, gt_masks=gt_masks,
                        train_rois_per_image=self.kwargs['train_rois_per_image'],
                        roi_pos_ratio=self.kwargs['roi_pos_ratio'],
                        num_classes=len(self.dataset.classes_dict.keys()),
                        bbox_std=self.kwargs['bbox_std'],
                        use_mini_mask=self.kwargs['use_mini_mask'],
                        mask_shape=self.kwargs['mask_shape'],
                        image_shape=self.kwargs['image_shape']
                    )

            # Init batch arrays
            if gen_batch == 0:
                batch_image_meta = np.zeros(
                    (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [self.batch_size, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [self.batch_size, self.kwargs['rpn_train_anchors_per_image'], 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (self.batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (self.batch_size, self.kwargs['max_gt_instances']), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (self.batch_size, self.kwargs['max_gt_instances'], 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (self.batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     self.kwargs['max_gt_instances']), dtype=gt_masks.dtype)
                if self.random_rois:
                    batch_rpn_rois = np.zeros(
                        (self.batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if self.detection_targets:
                        batch_rois = np.zeros(
                            (self.batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (self.batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (self.batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (self.batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

                if self.return_original:
                    batch_original_imgs = []
                    batch_original_masks = []
                    batch_original_class_ids = []
                    batch_original_bboxes = []

                    # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.kwargs['max_gt_instances']:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.kwargs['max_gt_instances'], replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to a batch
            batch_image_meta[gen_batch] = image_meta
            batch_rpn_match[gen_batch] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[gen_batch] = rpn_bbox
            batch_images[gen_batch] = image
            batch_gt_class_ids[gen_batch, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[gen_batch, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[gen_batch, :, :, :gt_masks.shape[-1]] = gt_masks
            if self.random_rois:
                batch_rpn_rois[gen_batch] = rpn_rois
                if self.detection_targets:
                    batch_rois[gen_batch] = rois
                    batch_mrcnn_class_ids[gen_batch] = mrcnn_class_ids
                    batch_mrcnn_bbox[gen_batch] = mrcnn_bbox
                    batch_mrcnn_mask[gen_batch] = mrcnn_mask

            if self.return_original:
                batch_original_imgs.append(original_image)
                batch_original_masks.append(original_masks_array)
                batch_original_class_ids.append(original_class_ids)
                batch_original_bboxes.append(original_bboxes)

            # Update info about batch size
            gen_batch += 1
            # Choose next index for the next image in batch or take the last image if one epoch is about to end.
            index = min(index + 1, len(self.indexes) - 1)

        inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
        outputs = []

        if self.random_rois:
            inputs.extend([batch_rpn_rois])
            if self.detection_targets:
                inputs.extend([batch_rois])
                # Keras requires that output and targets have the same number of dimensions
                batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
                outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

        if self.cast_output:
            inputs = [cast(x, 'float32') for x in inputs]
            outputs = [cast(x, 'float32') for x in outputs]

        if self.return_original:
            inputs.extend([batch_original_imgs, batch_original_masks, batch_original_class_ids, batch_original_bboxes])

        return inputs, outputs

    def __getitem__(self, i):
        inputs, outputs = self.generate_batch(i)
        return inputs, outputs

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        """
        Обновление порядка данных после каждой эпохи
        Returns: None
        """
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def get_input_preprocess(normalize=None):
    """
    Input preprocessing
    Args:
        normalize: dict, with normalization parameters: mean, std

    Returns: albumentations.Compose or None

    """
    test_transform = None
    if normalize:
        test_transform = [img_album.Normalize(mean=normalize['mean'],
                                              std=normalize['std'],
                                              max_pixel_value=255.0,
                                              always_apply=True)
                          ]
        test_transform = img_album.Compose(test_transform)

    return test_transform
