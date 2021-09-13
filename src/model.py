import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfl

from common import utils
from layers import losses
from layers import mrcnn_layers as mrcnnl


class FMaskRCNN(tf.keras.Model):
    def __init__(self, inputs, outputs, config, name='mask_rcnn', **kwargs):
        """
        tf.keras.Model for functional Mask-RCNN with overriden train and evaluation methods.
        Args:
            inputs:           Data inputs (x), list
            outputs           Data outputs (y), list
            config:           General MaskRCNN config, dict
            name:             Model name, string
            **kwargs:
        """
        super(FMaskRCNN, self).__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)
        self.config = config
        self.training = config['training']
        self.losses_dict = {}
        self.losses_tracker_dict = {}

    def compile(self, optimizer, losses_dict=None, run_eagerly=True, **kwargs):

        super(FMaskRCNN, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)
        # Update a dict of losses
        if losses_dict:
            self.losses_dict.update(losses_dict)
            # Add l2 regularization to losses
            self.losses_dict.update({'l2_regularizer': losses.L2RegLoss(model=self, config=self.config)})
            # Add losses tracker
            for loss_name in self.losses_dict.keys():
                self.losses_tracker_dict.update({loss_name: tf.keras.metrics.Mean(name=loss_name)})
            self.losses_tracker_dict.update({'loss_sum': tf.keras.metrics.Mean(name='loss_sum')})
        print(f'MaskRCNN Losses:\n' + ''.join([f'{name}: {loss}\n' for name, loss in self.losses_dict.items()]))

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch or at the start of `evaluate()`.
        # If you don't implement this property, you have to call `reset_states()` yourself at the time of your choosing.
        return [x for x in self.losses_tracker_dict.values()]

    def train_step(self, inputs):

        # Overriding training step
        if self.config['use_rpn_rois']:
            batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
            batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = inputs
        else:
            batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
            batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, random_rois = inputs

        with tf.GradientTape() as tape:
            outputs = self(inputs)
            rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, mrcnn_mask, \
            rpn_rois, active_class_ids, output_rois, target_class_ids, target_mask, target_bbox = outputs

            rpn_class_loss_val = self.losses_dict['rpn_class_loss'].call(rpn_match=batch_rpn_match,
                                                                         rpn_class_logits=rpn_class_logits)

            rpn_bbox_loss_val = self.losses_dict['rpn_bbox_loss'].call(target_bbox=batch_rpn_bbox,
                                                                       rpn_match=batch_rpn_match,
                                                                       rpn_bbox=rpn_bbox)

            mrcnn_class_loss_val = self.losses_dict['mrcnn_class_loss'].call(target_class_ids=target_class_ids,
                                                                             pred_class_logits=mrcnn_class_logits,
                                                                             active_class_ids=active_class_ids)

            mrcnn_bbox_loss_val = self.losses_dict['mrcnn_bbox_loss'].call(target_bbox=target_bbox,
                                                                           target_class_ids=target_class_ids,
                                                                           pred_bbox=mrcnn_bbox)

            mrcnn_mask_loss_val = self.losses_dict['mrcnn_mask_loss'].call(target_masks=target_mask,
                                                                           target_class_ids=target_class_ids,
                                                                           pred_masks=mrcnn_mask)

            l2_reg_loss_val = self.losses_dict['l2_regularizer'].call()

            # Assertions to prevent such anomalies as backprop with nans
            # It is possible because of some data augmentation operations
            assert not np.any(np.isnan(rpn_class_loss_val))
            assert not np.any(np.isnan(rpn_bbox_loss_val))
            assert not np.any(np.isnan(mrcnn_class_loss_val))
            assert not np.any(np.isnan(mrcnn_bbox_loss_val))
            assert not np.any(np.isnan(mrcnn_mask_loss_val))
            assert not np.any(np.isnan(l2_reg_loss_val))

            # Calculate summary loss with weights stored in self.config['loss_weights']
            loss = tf.tensordot(
                a=[rpn_class_loss_val, rpn_bbox_loss_val, mrcnn_class_loss_val,
                   mrcnn_bbox_loss_val, mrcnn_mask_loss_val],
                b=tf.cast(self.config['loss_weights'], 'float32'), axes=1)
            # Add L2-regularization to the total loss
            loss = tf.math.add(loss, l2_reg_loss_val)

        # Use the gradient tape to automatically retrieve the gradients of the trainable variables \
        # with respect to the loss.
        grads = tape.gradient(loss, self.trainable_weights)

        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        # If some grads do not exist< skip them
        # Check if need, None grads usually are for gamma, beta BatchNorm if l2 regularization skips them
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))
        self.optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(grads, self.trainable_variables) if grad is not None)

        # Update loss tracker
        for loss_name, loss_value in zip(['rpn_class_loss', 'rpn_bbox_loss', 'mrcnn_class_loss', 'mrcnn_bbox_loss',
                                          'mrcnn_mask_loss', 'l2_regularizer', 'loss_sum'
                                          ],
                                         [rpn_class_loss_val, rpn_bbox_loss_val, mrcnn_class_loss_val,
                                          mrcnn_bbox_loss_val, mrcnn_mask_loss_val, l2_reg_loss_val, loss]
                                         ):
            self.losses_tracker_dict[loss_name].update_state(loss_value)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):

        # Overriding test step
        if self.config['use_rpn_rois']:
            batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
            batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = inputs
        else:
            batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
            batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, random_rois = inputs

        outputs = self(inputs)
        rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, mrcnn_mask, \
        rpn_rois, active_class_ids, output_rois, target_class_ids, target_mask, target_bbox = outputs

        rpn_class_loss_val = self.losses_dict['rpn_class_loss'].call(rpn_match=batch_rpn_match,
                                                                     rpn_class_logits=rpn_class_logits)

        rpn_bbox_loss_val = self.losses_dict['rpn_bbox_loss'].call(target_bbox=batch_rpn_bbox,
                                                                   rpn_match=batch_rpn_match,
                                                                   rpn_bbox=rpn_bbox)

        mrcnn_class_loss_val = self.losses_dict['mrcnn_class_loss'].call(target_class_ids=target_class_ids,
                                                                         pred_class_logits=mrcnn_class_logits,
                                                                         active_class_ids=active_class_ids)

        mrcnn_bbox_loss_val = self.losses_dict['mrcnn_bbox_loss'].call(target_bbox=target_bbox,
                                                                       target_class_ids=target_class_ids,
                                                                       pred_bbox=mrcnn_bbox)

        mrcnn_mask_loss_val = self.losses_dict['mrcnn_mask_loss'].call(target_masks=target_mask,
                                                                       target_class_ids=target_class_ids,
                                                                       pred_masks=mrcnn_mask)

        # Calculate summary loss with weights stored in self.config['loss_weights']
        loss = tf.tensordot(
            a=[rpn_class_loss_val, rpn_bbox_loss_val, mrcnn_class_loss_val,
               mrcnn_bbox_loss_val, mrcnn_mask_loss_val],
            b=tf.cast(self.config['loss_weights'], 'float32'), axes=1)

        # Update loss tracker
        for loss_name, loss_value in zip(['rpn_class_loss', 'rpn_bbox_loss', 'mrcnn_class_loss', 'mrcnn_bbox_loss',
                                          'mrcnn_mask_loss', 'loss_sum'
                                          ],
                                         [rpn_class_loss_val, rpn_bbox_loss_val, mrcnn_class_loss_val,
                                          mrcnn_bbox_loss_val, mrcnn_mask_loss_val, loss]
                                         ):
            self.losses_tracker_dict[loss_name].update_state(loss_value)

        return {m.name: m.result() for m in self.metrics}


def mask_rcnn_functional(config):
    # Construct a model in functional API

    # Prevent creating keras names with index increment.
    # It is important for weights setting from training to inference graph
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    # Inputs
    input_image = tfl.Input(shape=[config['image_shape'][0],
                                   config['image_shape'][1],
                                   config['image_shape'][2]],
                            name="input_image")
    input_image_meta = tfl.Input(shape=config['meta_shape'], name="input_image_meta")

    if config['training']:
        print('[MaskRCNN] Training mode')
        # Regions Proposal Network Ground truth inputs
        input_rpn_match = tfl.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = tfl.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = tfl.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = tfl.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = mrcnnl.NormBoxesLayer()((input_gt_boxes, tf.shape(input_image)[1:3]))
        # 3. GT Masks (zero padded)
        # [batch, height, width, MAX_GT_INSTANCES]
        if config['use_mini_masks']:
            input_gt_masks = tfl.Input(shape=[config['mini_mask_shape'][0],
                                              config['mini_mask_shape'][1], None],
                                       name="input_gt_masks", dtype=bool)
        else:
            input_gt_masks = tfl.Input(shape=[config['image_shape'][0], config['image_shape'][1], None],
                                       name="input_gt_masks", dtype=bool)

        anchors = mrcnnl.AnchorsLayer(config=config, training=config['training'])(input_image)
        proposal_count = config['post_nms_rois_training']

    else:
        print('[MaskRCNN] Inference mode')
        # Anchors in normalized coordinates
        # input_anchors = tfl.Input(shape=[None, 4], name="input_anchors")
        anchors = mrcnnl.AnchorsLayer(config=config, training=config['training'])(input_image)  # input_anchors
        proposal_count = config['post_nms_rois_inference']

    print(f"""[MaskRCNN] Backbone architecture: {config['backbone']}""")
    backbone = mrcnnl.MaskRCNNBackbone(config=config).build()
    _, c2, c3, c4, c5 = backbone(input_image)

    # rpn_feature_maps =   [p2, p3, p4, p5, p6]
    # mrcnn_feature_maps = [p2, p3, p4, p5]
    rpn_feature_maps, mrcnn_feature_maps = mrcnnl.upsampling_graph(inputs=[c2, c3, c4, c5], config=config)

    # RPN Model
    rpn_model = mrcnnl.build_rpn_model(anchor_stride=config['rpn_anchor_stride'],
                                       anchors_per_location=len(config['rpn_anchor_ratios']),
                                       depth=config['top_down_pyramid_size'],
                                       training=config['training'])

    # Loop through pyramid layers
    layer_outputs = []
    for p in rpn_feature_maps:
        layer_outputs.append(rpn_model([p]))

    # Concatenate layer outputs
    # Convert from list of lists of level outputs to list of lists
    # of outputs across levels.
    # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [tfl.Concatenate(axis=1, name=f'concat_{n}')(list(o))
               for o, n in zip(outputs, output_names)]
    rpn_class_logits, rpn_class, rpn_bbox = outputs

    # Proposal layer
    rpn_rois = mrcnnl.ProposalLayer(proposal_count=proposal_count, config=config)([rpn_class, rpn_bbox, anchors])

    if config['training']:
        # Class ID mask to mark class IDs supported by the dataset the image came from.
        active_class_ids = tfl.Lambda(lambda x: utils.parse_image_meta_graph(x)["active_class_ids"],
                                      name='parse_image_meta_graph')(input_image_meta)

        if config['use_rpn_rois']:
            target_rois = rpn_rois
        else:
            # Ignore predicted ROIs and use ROIs provided as an input.
            input_rois = tfl.Input(shape=[config['post_nms_roi_training'], 4], name="input_roi", dtype=np.int32)
            # Normalize coordinates
            target_rois = mrcnnl.NormBoxesLayer(name='norm_boxes_layer')((input_rois, tf.shape(input_image)[1:3]))
        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox, target_mask = \
            mrcnnl.DetectionTargetLayer(config, name="detection_targets_layer")(
                [target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        # Network Heads
        mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = \
            mrcnnl.fpn_classifier_graph(inputs=[rois, input_image_meta, mrcnn_feature_maps],
                                        pool_size=config['pool_size'],
                                        fc_layers_size=config['fpn_cls_fc_layers_size'],
                                        num_classes=config['num_classes'],
                                        train_bn=config['train_bn'],
                                        batch_size=config['batch_size'],
                                        post_nms_rois_inference=config['post_nms_rois_inference'],
                                        training=config['training']
                                        )
        mrcnn_mask = mrcnnl.fpn_mask_graph(inputs=[rois, input_image_meta, mrcnn_feature_maps],
                                           pool_size=config['mask_pool_size'],
                                           num_classes=config['num_classes'],
                                           train_bn=config['train_bn'])

        # Model training inputs
        inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
                  input_gt_class_ids, input_gt_boxes, input_gt_masks]
        if not config['use_rpn_rois']:
            inputs.append(input_rois)

        # Model training outputs
        outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                   mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, mrcnn_mask,
                   rpn_rois, active_class_ids, rois, target_class_ids, target_mask, target_bbox]
        name = 'mask_rcnn_training'

    else:
        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = \
            mrcnnl.fpn_classifier_graph(inputs=[rpn_rois, input_image_meta, mrcnn_feature_maps],
                                        pool_size=config['pool_size'],
                                        fc_layers_size=config['fpn_cls_fc_layers_size'],
                                        num_classes=config['num_classes'],
                                        train_bn=config['train_bn'],
                                        batch_size=config['batch_size'],
                                        post_nms_rois_inference=config['post_nms_rois_inference'],
                                        training=config['training']
                                        )

        # Detections output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        detections = mrcnnl.DetectionLayer(proposals=proposal_count,
                                           batch_size=config['batch_size'],
                                           images_per_gpu=config['images_per_gpu'],
                                           detection_min_confidence=config['detection_min_confidence'],
                                           detection_max_instances=config['detection_max_instances'],
                                           detection_nms_threshold=config['detection_nms_threshold'],
                                           bbox_std_dev=config['bbox_std_dev'],
                                           )(
            [rpn_rois, mrcnn_probs, mrcnn_bbox, input_image_meta])
        # Create masks for detections
        detection_boxes = mrcnnl.DetectedBoxesExtraction()(detections)
        mrcnn_mask = mrcnnl.fpn_mask_graph(inputs=[detection_boxes, input_image_meta, mrcnn_feature_maps],
                                           pool_size=config['mask_pool_size'],
                                           num_classes=config['num_classes'],
                                           train_bn=config['train_bn'])

        # Model inference inputs and outputs
        inputs = [input_image, input_image_meta]
        outputs = [detections, mrcnn_probs, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]
        name = 'mask_rcnn_inference'

    # Make model graph in a customized tf.keras.Model with overridden train and test steps
    model = FMaskRCNN(inputs=inputs, outputs=outputs, config=config, name=name)
    return model


class MaskRCNN(tf.keras.Model):

    def __init__(self, config, name='mask_rcnn', **kwargs):
        """
        Subclassed version of MaskRCNN.
        Such complex subclassed model can not be saved in HDF5 for now, but is useful for research and debug.
        Args:
            config:      General MaskRCNN config, dict
            name:        Model name, string
            **kwargs:
        """
        super(MaskRCNN, self).__init__(name=name, **kwargs)
        self.config = config
        self.training = config['training']
        self.losses_dict = {}
        self.losses_tracker_dict = {}

        # Backbone model for features extraction
        # self.backbone = mrcnnl.ResNetLayer(resnet_type=self.config['backbone'], stage5=True, train_bn=self.training)
        self.backbone = mrcnnl.MaskRCNNBackbone(config=config).build()

        # Upsampling layer
        self.upsampling_layer = mrcnnl.UpSamplingLayer(config=self.config)

        # Anchors
        self.anchors_layer = mrcnnl.AnchorsLayer(config=config, training=self.training)
        self.anchors = self.anchors_layer(None)

        # RPN model
        self.rpn_layer = mrcnnl.RPNLayer(anchors_per_location=len(config['rpn_anchor_ratios']),
                                         anchor_stride=config['rpn_anchor_stride'])

        if self.training:
            # Generate proposals
            # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates and zero padded.
            self.proposal_count = config['post_nms_rois_training']
            # Class ID mask to mark class IDs supported by the dataset the image came from.
            self.parse_img_meta_layer = mrcnnl.ImageMetaLayer()
            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            self.detection_target_layer = mrcnnl.DetectionTargetLayer(self.config)
            # Coordinates normalization layer
            self.norm_boxes_layer = mrcnnl.NormBoxesLayer()
        else:
            self.proposal_count = config['post_nms_rois_inference']
            self.detections_layer = mrcnnl.DetectionLayer(proposals=self.proposal_count,
                                                          batch_size=config['batch_size'],
                                                          images_per_gpu=config['images_per_gpu'],
                                                          detection_min_confidence=config['detection_min_confidence'],
                                                          detection_max_instances=config['detection_max_instances'],
                                                          detection_nms_threshold=config['detection_nms_threshold'],
                                                          bbox_std_dev=config['bbox_std_dev'], )
            self.detected_boxes_extraction = mrcnnl.DetectedBoxesExtraction()

        # Proposals layer
        self.proposal_layer = mrcnnl.ProposalLayer(proposal_count=self.proposal_count, config=config)

        # FPN classifier layer
        self.fpn_classifier = \
            mrcnnl.FPNClassifier(pool_size=self.config['pool_size'],
                                 fc_layers_size=self.config['fpn_cls_fc_layers_size'],
                                 num_classes=self.config['num_classes'],
                                 train_bn=self.config['train_bn'])

        # FPN mask layer
        self.fpn_mask_layer = \
            mrcnnl.FPNMaskLayer(pool_size=self.config['mask_pool_size'],
                                num_classes=self.config['num_classes'],
                                train_bn=self.config['train_bn'])

        # Build subclassed model
        self.call([np.random.uniform(size=x) for x in [(self.config['batch_size'], 512, 512, 3),
                                                       (self.config['batch_size'], 14),
                                                       (self.config['batch_size'], 65472, 1),
                                                       (self.config['batch_size'], 256, 4),
                                                       (self.config['batch_size'], 100),
                                                       (self.config['batch_size'], 100, 4),
                                                       (self.config['batch_size'], 512, 512, 100),
                                                       ]
                   ]
                  )

    def compile(self, optimizer, losses_dict=None, run_eagerly=True, **kwargs):

        super(MaskRCNN, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)
        # Update a dict of losses
        if losses_dict:
            self.losses_dict.update(losses_dict)
            # Add l2 regularization to losses
            self.losses_dict.update({'l2_regularizer': losses.L2RegLoss(model=self, config=self.config)})
            # Add losses tracker
            for loss_name in self.losses_dict.keys():
                self.losses_tracker_dict.update({loss_name: tf.keras.metrics.Mean(name=loss_name)})
            self.losses_tracker_dict.update({'loss_sum': tf.keras.metrics.Mean(name='loss_sum')})
        print(f'MaskRCNN Losses:\n' + ''.join([f'{name}: {loss}\n' for name, loss in self.losses_dict.items()]))

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch or at the start of `evaluate()`.
        # If you don't implement this property, you have to call `reset_states()` yourself at the time of your choosing.
        return [x for x in self.losses_tracker_dict.values()]

    def call(self, inputs, **kwargs):

        if self.training:
            # Training input
            if self.config['use_rpn_rois']:
                batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
                batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = inputs
            else:
                batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
                batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, random_rois = inputs
        else:
            # Inference input
            # Input for inference requires preprocessing
            batch_images, batch_images_meta = inputs

        # Extract c2, c3, c4, c5 features
        _, c2, c3, c4, c5 = self.backbone(batch_images)

        # rpn_feature_maps = [p2, p3, p4, p5, p6]
        # mrcnn_feature_maps = [p2, p3, p4, p5]
        rpn_feature_maps, mrcnn_feature_maps = self.upsampling_layer([c2, c3, c4, c5])

        # Loop through pyramid layers
        layer_outputs = []  # A list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn_layer(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [tfl.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        rpn_rois = self.proposal_layer([rpn_class, rpn_bbox, self.anchors])

        if self.training:
            batch_gt_boxes = self.norm_boxes_layer((batch_gt_boxes, tf.shape(batch_images)[1: 3]))

            active_class_ids = self.parse_img_meta_layer(batch_images_meta)['active_class_ids']

            if self.config['use_rpn_rois']:
                target_rois = rpn_rois
            else:
                target_rois = self.norm_boxes_layer((random_rois, tf.shape(batch_images)[1: 3]))

            rois, target_class_ids, target_bbox, target_mask = \
                self.detection_target_layer([target_rois, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks])

            # Network heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                self.fpn_classifier([rois, batch_images_meta, mrcnn_feature_maps])
            mrcnn_mask = self.fpn_mask_layer([rois, batch_images_meta, mrcnn_feature_maps])
            output_rois = tf.identity(rois)

            return rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
                   rpn_rois, active_class_ids, output_rois, target_class_ids, target_mask, target_bbox

        else:
            # Network heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                self.fpn_classifier([rpn_rois, batch_images_meta, mrcnn_feature_maps])
            # Detections output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
            detections = self.detections_layer([rpn_rois, mrcnn_class, mrcnn_bbox, batch_images_meta])
            # Create masks for detections
            detection_boxes = self.detected_boxes_extraction(detections)
            mrcnn_mask = self.fpn_mask_layer([detection_boxes, batch_images_meta, mrcnn_feature_maps])

            return detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox

    def train_step(self, data):
        # Overriding  training step
        # In general, batch consists of inputs and outputs. Will be here for a while in case of adding func api
        inputs = data
        if self.config['use_rpn_rois']:
            batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
            batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = inputs
        else:
            batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
            batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, random_rois = inputs

        with tf.GradientTape() as tape:
            outputs = self(inputs)
            rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
            rpn_rois, active_class_ids, output_rois, target_class_ids, target_mask, target_bbox = outputs

            rpn_class_loss_val = self.losses_dict['rpn_class_loss'].call(rpn_match=batch_rpn_match,
                                                                         rpn_class_logits=rpn_class_logits)

            rpn_bbox_loss_val = self.losses_dict['rpn_bbox_loss'].call(target_bbox=batch_rpn_bbox,
                                                                       rpn_match=batch_rpn_match,
                                                                       rpn_bbox=rpn_bbox)

            mrcnn_class_loss_val = self.losses_dict['mrcnn_class_loss'].call(target_class_ids=target_class_ids,
                                                                             pred_class_logits=mrcnn_class_logits,
                                                                             active_class_ids=active_class_ids)

            mrcnn_bbox_loss_val = self.losses_dict['mrcnn_bbox_loss'].call(target_bbox=target_bbox,
                                                                           target_class_ids=target_class_ids,
                                                                           pred_bbox=mrcnn_bbox)

            mrcnn_mask_loss_val = self.losses_dict['mrcnn_mask_loss'].call(target_masks=target_mask,
                                                                           target_class_ids=target_class_ids,
                                                                           pred_masks=mrcnn_mask)

            l2_reg_loss_val = self.losses_dict['l2_regularizer'].call()

            # Assertions to prevent backprop with nans
            assert not np.any(np.isnan(rpn_class_loss_val))
            assert not np.any(np.isnan(rpn_bbox_loss_val))
            assert not np.any(np.isnan(mrcnn_class_loss_val))
            assert not np.any(np.isnan(mrcnn_bbox_loss_val))
            assert not np.any(np.isnan(mrcnn_mask_loss_val))
            assert not np.any(np.isnan(l2_reg_loss_val))

            # Calculate summary loss with weights stored in self.config['loss_weights']
            loss = tf.tensordot(
                a=[rpn_class_loss_val, rpn_bbox_loss_val, mrcnn_class_loss_val,
                   mrcnn_bbox_loss_val, mrcnn_mask_loss_val],
                b=tf.cast(self.config['loss_weights'], 'float32'), axes=1)
            loss += l2_reg_loss_val

        # Use the gradient tape to automatically retrieve the gradients of the trainable variables \
        # with respect to the loss.
        grads = tape.gradient(loss, self.trainable_weights)

        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        # If some grads do not exist, skip them
        # Check if need, None grads usually are for gamma, beta BatchNorm if l2 regularization skips them
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))
        self.optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(grads, self.trainable_variables) if grad is not None)

        # Update loss tracker
        for loss_name, loss_value in zip(['rpn_class_loss', 'rpn_bbox_loss', 'mrcnn_class_loss', 'mrcnn_bbox_loss',
                                          'mrcnn_mask_loss', 'l2_regularizer', 'loss_sum'
                                          ],
                                         [rpn_class_loss_val, rpn_bbox_loss_val, mrcnn_class_loss_val,
                                          mrcnn_bbox_loss_val, mrcnn_mask_loss_val, l2_reg_loss_val, loss]
                                         ):
            self.losses_tracker_dict[loss_name].update_state(loss_value)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Overriding evaluation step
        # In general, batch consists of inputs and outputs. Will be here for a while in case of adding func api
        inputs = data
        if self.config['use_rpn_rois']:
            batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
            batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = inputs
        else:
            batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
            batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, random_rois = inputs

        outputs = self(inputs)
        rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
        rpn_rois, active_class_ids, output_rois, target_class_ids, target_mask, target_bbox = outputs

        rpn_class_loss_val = self.losses_dict['rpn_class_loss'].call(rpn_match=batch_rpn_match,
                                                                     rpn_class_logits=rpn_class_logits)

        rpn_bbox_loss_val = self.losses_dict['rpn_bbox_loss'].call(target_bbox=batch_rpn_bbox,
                                                                   rpn_match=batch_rpn_match,
                                                                   rpn_bbox=rpn_bbox)

        mrcnn_class_loss_val = self.losses_dict['mrcnn_class_loss'].call(target_class_ids=target_class_ids,
                                                                         pred_class_logits=mrcnn_class_logits,
                                                                         active_class_ids=active_class_ids)

        mrcnn_bbox_loss_val = self.losses_dict['mrcnn_bbox_loss'].call(target_bbox=target_bbox,
                                                                       target_class_ids=target_class_ids,
                                                                       pred_bbox=mrcnn_bbox)

        mrcnn_mask_loss_val = self.losses_dict['mrcnn_mask_loss'].call(target_masks=target_mask,
                                                                       target_class_ids=target_class_ids,
                                                                       pred_masks=mrcnn_mask)

        # Calculate summary loss with weights stored in self.config['loss_weights']
        loss = tf.tensordot(
            a=[rpn_class_loss_val, rpn_bbox_loss_val, mrcnn_class_loss_val,
               mrcnn_bbox_loss_val, mrcnn_mask_loss_val],
            b=tf.cast(self.config['loss_weights'], 'float32'), axes=1)

        # Update loss tracker
        for loss_name, loss_value in zip(['rpn_class_loss', 'rpn_bbox_loss', 'mrcnn_class_loss', 'mrcnn_bbox_loss',
                                          'mrcnn_mask_loss', 'loss_sum'
                                          ],
                                         [rpn_class_loss_val, rpn_bbox_loss_val, mrcnn_class_loss_val,
                                          mrcnn_bbox_loss_val, mrcnn_mask_loss_val, loss]
                                         ):
            self.losses_tracker_dict[loss_name].update_state(loss_value)

        return {m.name: m.result() for m in self.metrics}
