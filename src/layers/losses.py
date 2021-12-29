import tensorflow as tf


# Losses in subclassed API
class RPNClassLoss(tf.keras.losses.Loss):

    def __init__(self, name="rpn_class_loss", **kwargs):
        """
        RPN anchor classifier loss.
        Args:
            name: rpn_class_loss
        """
        self.name = name
        super(RPNClassLoss, self).__init__(name=name, **kwargs)

    def call(self, rpn_match, rpn_class_logits, **kwargs) -> dict:
        """RPN anchor classifier loss.

        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                   -1=negative, 0=neutral anchor.
        rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
        """
        # Squeeze last dim to simplify
        rpn_match = tf.squeeze(rpn_match, -1)
        # Get anchor classes. Convert the -1/+1 match to 0/1 values.
        anchor_class = tf.cast(tf.math.equal(rpn_match, 1), tf.int32)
        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors (match value = 0) don't.
        indices = tf.where(tf.math.not_equal(rpn_match, 0))
        # Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        # Cross entropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=anchor_class, y_pred=rpn_class_logits,
                                                               from_logits=True)
        loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.math.reduce_mean(loss), tf.constant(0.0))
        return loss


class RPNBboxLoss(tf.keras.losses.Loss):

    def __init__(self, images_per_gpu, name="rpn_bbox_loss", **kwargs):
        """
        Return the RPN bounding box loss graph.
        Args:
            images_per_gpu:
            name: rpn_bbox_loss
        """
        self.name = name
        self.images_per_gpu = images_per_gpu
        super(RPNBboxLoss, self).__init__(name=name, **kwargs)

    def batch_pack_graph(self, x, counts):
        """Picks different number of values from each row
        in x depending on the values in counts.
        """
        outputs = []
        for i in range(self.images_per_gpu):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)

    def smooth_l1_loss(self, y_true, y_pred):
        """Implements Smooth-L1 loss.
        y_true and y_pred are typically: [N, 4], but could be any shape.
        """
        diff = tf.math.abs(y_true - y_pred)
        less_than_one = tf.cast(tf.math.less(diff, 1.0), "float32")
        loss = less_than_one * (0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)
        return loss

    def call(self, target_bbox, rpn_match, rpn_bbox, **kwargs):
        """Return the RPN bounding box loss graph.

        config: the model config object.
        target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed bbox deltas.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                   -1=negative, 0=neutral anchor.
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        """
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        rpn_match = tf.squeeze(rpn_match, -1)
        indices = tf.where(tf.math.equal(rpn_match, 1))

        # Pick bbox deltas that contribute to the loss
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)  # (3,4) (4, 4)

        # Trim target bounding box deltas to the same length as rpn_bbox.
        batch_counts = tf.math.reduce_sum(tf.cast(tf.math.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = self.batch_pack_graph(target_bbox, batch_counts)

        loss = self.smooth_l1_loss(target_bbox, rpn_bbox)

        loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.math.reduce_mean(loss), tf.constant(0.0))
        return loss


class MRCNNClassLoss(tf.keras.losses.Loss):

    def __init__(self, batch_size, name="mrcnn_class_loss", **kwargs):
        """
        Loss for the classifier head of Mask RCNN.
        Args:
            name: mrcnn_class_loss
        """
        self.name = name
        self.batch_size = batch_size
        super(MRCNNClassLoss, self).__init__(name=name, **kwargs)

    def call(self, target_class_ids, pred_class_logits, active_class_ids, eps=1e-5, **kwargs):
        """Loss for the classifier head of Mask RCNN.

        target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
            padding to fill in the array.
        pred_class_logits: [batch, num_rois, num_classes]
        active_class_ids: [batch, num_classes]. Has a value of 1 for
            classes that are in the dataset of the image, and 0 for classes that are not in the dataset.
            The position of ones and zeros means the class index.
        """

        target_class_ids = tf.cast(target_class_ids, 'int64')

        # Find predictions of classes that are not in the dataset.
        pred_class_ids = tf.argmax(pred_class_logits, axis=2)
        pred_active = tf.stack([tf.gather(active_class_ids[b], pred_class_ids[b]) for b in range(self.batch_size)],
            axis=0)
        # Loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)

        # Erase losses of predictions of classes that are not in the active classes of the image.
        loss = loss * pred_active

        # Compute loss mean. Use only predictions that contribute to the loss to get a correct mean.
        loss = tf.math.reduce_sum(loss) / (tf.math.reduce_sum(pred_active) + tf.constant(eps))
        return loss


class MRCNNBboxLoss(tf.keras.losses.Loss):

    def __init__(self, num_classes, name='mrcnn_bbox_loss', **kwargs):
        """
        Loss for Mask R-CNN bounding box refinement.
        Args:
            name: mrcnn_bbox_loss
        """
        self.name = name
        self.num_classes = num_classes
        super(MRCNNBboxLoss, self).__init__(name=name, **kwargs)

    def smooth_l1_loss(self, y_true, y_pred):
        """Implements Smooth-L1 loss.
        y_true and y_pred are typically: [N, 4], but could be any shape.
        """
        diff = tf.math.abs(y_true - y_pred)
        less_than_one = tf.cast(tf.math.less(diff, 1.0), "float32")
        loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
        return loss

    def call(self, target_bbox, target_class_ids, pred_bbox, **kwargs):
        """Loss for Mask R-CNN bounding box refinement.

        target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
        target_class_ids: [batch, num_rois]. Integer class IDs.
        pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        """
        # Reshape to merge batch and roi dimensions for simplicity.
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        target_bbox = tf.reshape(target_bbox, (-1, 4))
        pred_bbox = tf.reshape(pred_bbox, (-1, self.num_classes, 4))

        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indices.
        positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = tf.gather(target_bbox, positive_roi_ix)
        pred_bbox = tf.gather_nd(pred_bbox, indices)

        # Smooth-L1 Loss
        loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                                       tf.math.reduce_mean(self.smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox)),
                                       tf.constant(0.0))

        return loss


class MRCNNMaskLoss(tf.keras.losses.Loss):

    def __init__(self, name='mrcnn_mask_loss', **kwargs):
        """
        Mask binary cross-entropy loss for the masks head.
        Args:
            name: mrcnn_mask_loss
        """
        self.name = name
        super(MRCNNMaskLoss, self).__init__(name=name, **kwargs)

    def call(self, target_masks, target_class_ids, pred_masks, **kwargs):
        """Mask binary cross-entropy loss for the masks head.

        target_masks: [batch, num_rois, height, width].
            A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                    with values from 0 to 1.
        """
        # Reshape for simplicity. Merge first two dimensions into one.
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        mask_shape = tf.shape(target_masks)
        target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(pred_masks)
        pred_masks = tf.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
        # Permute predicted masks to [N, num_classes, height, width]
        pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
        indices = tf.stack([positive_ix, positive_class_ids], axis=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = tf.gather(target_masks, positive_ix)
        y_pred = tf.gather_nd(pred_masks, indices)

        # Compute binary cross entropy. If no positive ROIs, then return 0.
        # shape: [batch, roi, num_classes]
        loss = tf.keras.backend.switch(tf.size(y_true) > 0,
                                       tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred),
                                       tf.constant(0.0))
        loss = tf.math.reduce_mean(loss)
        return loss


class L2RegLoss(tf.keras.losses.Loss):
    def __init__(self, model, config, name='l2_regularizer', **kwargs):
        super(L2RegLoss, self).__init__(name=name, **kwargs)
        self.name = name
        self.config = config
        self.model = model
        self.regularizer = tf.keras.regularizers.l2(self.config['weight_decay'])

    def call(self, dummy=None, **kwargs):
        # Skip gamma and beta weights of batch normalization layers.
        # Also skip biases from being regularized
        if self.config['l2_reg_batchnorm']:
            reg_losses = [self.regularizer(w) / tf.cast(tf.size(w), tf.float32) for w in self.model.trainable_weights]
        else:
            reg_losses = [self.regularizer(w) / tf.cast(tf.size(w), tf.float32) for w in self.model.trainable_weights if
                          'gamma' not in w.name and 'beta' not in w.name]

        loss = tf.add_n(reg_losses)
        return loss


# Losses in functional API
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.math.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.math.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.math.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.math.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=anchor_class, y_pred=rpn_class_logits,
                                                           from_logits=True)
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.math.reduce_mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(target_bbox, rpn_match, rpn_bbox, config):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(tf.math.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = tf.math.reduce_sum(tf.cast(tf.math.equal(rpn_match, 1), tf.int32), axis=1)

    def batch_pack_graph(x, counts, images_per_gpu):
        """Picks different number of values from each row
        in x depending on the values in counts.
        """
        outputs = []
        for i in range(images_per_gpu):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)

    target_bbox = batch_pack_graph(target_bbox, batch_counts, config['images_per_gpu'])
    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.math.reduce_mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids, config):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0 for classes that are not in the dataset.
        The position of ones and zeros means the class index.
    """

    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_active = tf.stack([tf.gather(active_class_ids[b], pred_class_ids[b]) for b in range(config['batch_size'])],
        axis=0)
    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox, config):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox, (-1, config['num_classes'], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                                   tf.math.reduce_mean(smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox)),
                                   tf.constant(0.0))
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = tf.keras.backend.switch(tf.size(y_true) > 0,
                                   tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred), tf.constant(0.0))
    loss = tf.math.reduce_mean(loss)
    return loss
