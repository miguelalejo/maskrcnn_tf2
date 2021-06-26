import cv2
import numpy as np
import tensorflow as tf
from common.utils import resize_image, compose_image_meta
from layers import losses
from model import mask_rcnn_functional
from training import get_optimizer


def process_input(input_image, config, preprocess_transform=lambda x: x / 255):
    """
    Process image for MaskRCNN inference
    Args:
        input_image:           Image read by cv2, array
        config:                MaskRCNN config, dict
        preprocess_transform:  Last image transformation before model inference
                               Default is scaling 1/255, function

    Returns: processed image

    """

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    original_input_shape = input_image.shape

    # Resize image
    resized_image, window, scale, padding, crop = resize_image(input_image,
                                                               min_dim=config['image_min_dim'],
                                                               min_scale=config['image_min_scale'],
                                                               max_dim=config['image_max_dim'],
                                                               mode=config['image_resize_mode'])
    # Get image meta
    image_meta = compose_image_meta(image_id=0,
                                    original_image_shape=original_input_shape,
                                    window=window,
                                    scale=scale,
                                    active_class_ids=np.zeros([config['num_classes']], dtype=np.int32),
                                    config=config)
    # Image transformation
    if preprocess_transform:
        resized_image = preprocess_transform(resized_image)
    return resized_image, image_meta, window


def set_backbone_weights(training_model, inference_model, verbose=False):
    """
    Set backbone weights from training to inference graph
    Args:
        training_model:   MaskRCNN training graph, tf.keras.Model
        inference_model:  MaskRCNN inference graph, tf.keras.Model
        verbose:          Print layers that get weights, bool

    Returns: inference_model

    """

    # Get backbone layer name
    _backbone_name = [x.name for x in training_model.layers if 'backbone' in x.name][0]
    training_bbone = training_model.get_layer(_backbone_name)

    for layer in training_bbone.layers:
        # Get backbone weights from training graph
        layer_name = layer.name
        layer_weights = layer.get_weights()

        if 'input' in layer_name:
            continue

        # Set weights to backbone in inference graph
        inference_model.get_layer(_backbone_name).get_layer(layer_name).set_weights(layer_weights)

        if verbose:
            print(f'Set weights: {layer_name}')

    return inference_model


def set_fpn_weights(training_model, inference_model, verbose=False):
    """
    Set feature pyramid network (FPN) weights from training to inference graph
    Args:
        training_model:   MaskRCNN training graph, tf.keras.Model
        inference_model:  MaskRCNN inference graph, tf.keras.Model
        verbose:          Print layers that get weights, bool

    Returns: inference_model

    """
    fpn_layers = ['fpn_c5p5', 'fpn_c4p4', 'fpn_c3p3', 'fpn_c2p2',
                  'fpn_p5', 'fpn_p4', 'fpn_p3', 'fpn_p2',
                  ]
    for layer_name in fpn_layers:
        # Get weights from training graph
        layer_weights = training_model.get_layer(layer_name).get_weights()
        # Set weights in inference graph
        inference_model.get_layer(layer_name).set_weights(layer_weights)

        if verbose:
            print(f'Set weights: {layer_name}')

    return inference_model


def set_rpn_weights(training_model, inference_model, verbose=False):
    """
    Set region proposal network (RPN) weights from training to inference graph
    Args:
        training_model:   MaskRCNN training graph, tf.keras.Model
        inference_model:  MaskRCNN inference graph, tf.keras.Model
        verbose:          Print layers that get weights, bool

    Returns: inference_model

    """
    training_rpn = training_model.get_layer('rpn_model')

    for layer in training_rpn.layers:

        layer_name = layer.name
        if 'input' in layer_name:
            continue
        # Get weights from training graph
        layer_weights = layer.get_weights()
        if len(layer_weights) == 0:
            print('Skipped zero-weights layer: ', layer_name)
            continue
        # Set weights in inference graph
        inference_model.get_layer('rpn_model').get_layer(layer_name).set_weights(layer_weights)

        if verbose:
            print(f'Set weights: {layer_name}')

    return inference_model


def set_mrcnn_head_weights(training_model, inference_model, verbose=False):
    """
    Set MaskRCNN head weights from training to inference graph
    Args:
        training_model:   MaskRCNN training graph,  tf.keras.Model
        inference_model:  MaskRCNN inference graph, tf.keras.Model
        verbose:          Print layers that get weights, bool

    Returns: inference_model

    """
    mrcnn_head_layers = ['mrcnn_mask_conv1', 'mrcnn_mask_bn1',
                         'mrcnn_mask_conv2', 'mrcnn_mask_bn2',
                         'mrcnn_mask_conv3', 'mrcnn_mask_bn3',
                         'mrcnn_mask_conv4', 'mrcnn_mask_bn4',

                         'mrcnn_class_conv1', 'mrcnn_class_bn1',
                         'mrcnn_class_conv2', 'mrcnn_class_bn2',

                         'fpnclf_mrcnn_class_logits', 'fpnclf_mrcnn_bbox_fc',

                         'mrcnn_mask_deconv', 'mrcnn_mask',
                         ]

    for layer_name in mrcnn_head_layers:

        # Get weights from training graph
        layer_weights = training_model.get_layer(layer_name).get_weights()
        # Set weights in inference graph
        inference_model.get_layer(layer_name).set_weights(layer_weights)

        if verbose:
            print(f'Set weights: {layer_name}')

    return inference_model


def weights_transfer(training_graph, inference_graph, verbose):
    """
    Transfer necessary weights from training MaskRCNN graph to inference MaskRCNN graph
    Args:
        training_graph:  MaskRCNN tf.keras.Model built with training=True
        inference_graph: MaskRCNN tf.keras.Model built with training=False
        verbose:         Print layers that get weights, bool

    Returns: inference_graph with weights set from training graph

    """
    inference_graph = set_backbone_weights(training_graph, inference_graph, verbose)
    inference_graph = set_fpn_weights(training_graph, inference_graph, verbose)
    inference_graph = set_rpn_weights(training_graph, inference_graph, verbose)
    inference_graph = set_mrcnn_head_weights(training_graph, inference_graph, verbose)

    return inference_graph


def load_mrcnn_weights(model, weights_path, verbose=True):
    """

    Args:
        model: MaskRCNN model in subclassed or functional API
        weights_path: path to model weights
        verbose: Print layers that get weights in inference mode, bool
    Returns: model

    """
    # It is necessary for making proper onnx graph
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    config = model.config

    if config['training']:
        print('\nWeights will be loaded to training graph\n')
        # Compile training model and load weights
        optimizer = get_optimizer(config['optimizer_kwargs'])
        losses_dict = {'rpn_class_loss': losses.RPNClassLoss(),
                       'rpn_bbox_loss': losses.RPNBboxLoss(images_per_gpu=config['images_per_gpu']),
                       'mrcnn_class_loss': losses.MRCNNClassLoss(batch_size=config['batch_size']),
                       'mrcnn_bbox_loss': losses.MRCNNBboxLoss(num_classes=config['num_classes']),
                       'mrcnn_mask_loss': losses.MRCNNMaskLoss(),
                       }
        model.compile(optimizer=optimizer, losses_dict=losses_dict, run_eagerly=True)
        # Load training model checkpoint
        model.load_weights(weights_path)
        print(model.summary())
    else:
        print('\nWeights for inference graph will be transferred from training graph\n')

        # Get training model
        _training_config = config
        _training_config.update({'training': True})
        training_model = mask_rcnn_functional(config=_training_config)
        # Compile training model and load weights
        optimizer = get_optimizer(config['optimizer_kwargs'])
        losses_dict = {'rpn_class_loss': losses.RPNClassLoss(),
                       'rpn_bbox_loss': losses.RPNBboxLoss(images_per_gpu=config['images_per_gpu']),
                       'mrcnn_class_loss': losses.MRCNNClassLoss(batch_size=config['batch_size']),
                       'mrcnn_bbox_loss': losses.MRCNNBboxLoss(num_classes=config['num_classes']),
                       'mrcnn_mask_loss': losses.MRCNNMaskLoss(),
                       }
        training_model.compile(optimizer=optimizer, losses_dict=losses_dict, run_eagerly=True)
        # Load training model checkpoint
        training_model.load_weights(weights_path)
        # Transfer weights from training to inference graph
        model = weights_transfer(training_graph=training_model,
                                 inference_graph=model,
                                 verbose=verbose)
        print(model.summary())
    return model
