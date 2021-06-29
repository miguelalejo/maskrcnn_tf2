import efficientnet.keras as efn
import numpy as np
import tensorflow as tf
from classification_models.keras import Classifiers
from common import utils
from tensorflow.keras import layers as tfl


# Subclassed tf.keras API
@tf.keras.utils.register_keras_serializable()
class NormBoxesLayer(tfl.Layer):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """

    def __init__(self, name='norm_boxes', **kwargs):
        super(NormBoxesLayer, self).__init__(name=name, **kwargs)
        self.shift = np.array((0., 0., 1., 1.))
        self.const = np.array(1.0)

    def build(self, input_shape):
        self.built = True
        super(NormBoxesLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # assert inputs is tuple
        boxes, shape = inputs
        h, w = tf.split(tf.cast(shape, tf.float32), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - self.const
        return tf.math.divide(boxes - self.shift, scale)

    def get_config(self):
        config = super(NormBoxesLayer, self).get_config()
        config.update({"shift": self.shift, 'const': self.const})
        return config


@tf.keras.utils.register_keras_serializable()
class ResnetConvBlock(tfl.Layer):
    def __init__(self, filters, kernel_size=3, strides=(2, 2), use_bias=True,
                 train_bn=True, name='resnet_conv_block', **kwargs):
        """
        Args:
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters:     list of integers, the nb_filters of 3 conv layer at main path
            strides:     strides for shortcut
            use_bias:    Boolean. To use or not use a bias in conv layers.
            name:        block name
            **kwargs:

        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        super(ResnetConvBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        filters1, filters2, filters3 = self.filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, kernel_size=(1, 1), strides=strides, use_bias=use_bias)
        self.bn2a = tf.keras.layers.BatchNormalization(trainable=train_bn)

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size=kernel_size, padding='same', use_bias=use_bias)
        self.bn2b = tf.keras.layers.BatchNormalization(trainable=train_bn)

        self.conv2c = tf.keras.layers.Conv2D(filters3, kernel_size=(1, 1), use_bias=use_bias)
        self.bn2c = tf.keras.layers.BatchNormalization(trainable=train_bn)

        self.conv_shortcut = tf.keras.layers.Conv2D(filters3, kernel_size=(1, 1), strides=strides, use_bias=use_bias)
        self.bn2_sc = tf.keras.layers.BatchNormalization(trainable=train_bn)

    def build(self, input_shape):
        self.built = True
        super(ResnetConvBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x)

        shortcut = self.conv_shortcut(inputs)
        shortcut = self.bn2_sc(shortcut)

        x += shortcut
        return tf.nn.relu(x)

    def get_config(self):
        config = super(ResnetConvBlock, self).get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'strides': self.strides})
        return config


@tf.keras.utils.register_keras_serializable()
class ResnetConvBlockSmall(tfl.Layer):

    def __init__(self, filter_size, kernel_size=3, strides=(2, 2), use_bias=True,
                 train_bn=True, name='resnet_conv_block_small', **kwargs):
        """
        Block for resnet18 and resnet34
        Args:
            filter_size:
            kernel_size:
            strides:
            use_bias:
            train_bn:
            name:
            **kwargs:
        """
        super(ResnetConvBlockSmall, self).__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv2a = tf.keras.layers.Conv2D(filters=self.filter_size, kernel_size=self.kernel_size, strides=strides,
                                             use_bias=use_bias, padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization(trainable=train_bn)

        self.conv2b = tf.keras.layers.Conv2D(filters=self.filter_size, kernel_size=self.kernel_size,
                                             use_bias=use_bias, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization(trainable=train_bn)

        self.conv_shortcut = tf.keras.layers.Conv2D(filters=self.filter_size, kernel_size=(1, 1), strides=strides,
                                                    use_bias=use_bias)
        self.bn2_sc = tf.keras.layers.BatchNormalization(trainable=train_bn)

    def build(self, input_shape):
        self.built = True
        super(ResnetConvBlockSmall, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = tf.nn.relu(x)

        shortcut = self.conv_shortcut(inputs)
        shortcut = self.bn2_sc(shortcut)

        x += shortcut
        return tf.nn.relu(x)

    def get_config(self):
        config = super(ResnetConvBlockSmall, self).get_config()
        config.update({'filter_size': self.filter_size, 'kernel_size': self.kernel_size, 'strides': self.strides})
        return config


@tf.keras.utils.register_keras_serializable()
class ResnetIdentityBlock(tfl.Layer):
    def __init__(self, filters, kernel_size=3, use_bias=True,
                 train_bn=True, name='resnet_identity_block', **kwargs):
        """
         Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
         And the shortcut should have subsample=(2,2) as well
        Args:
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters:     list of integers, the nb_filters of 3 conv layer at main path
            use_bias:    Boolean. To use or not use a bias in conv layers.
            name:        block name
            **kwargs:
        """
        super(ResnetIdentityBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        filters1, filters2, filters3 = self.filters

        self.conv2a = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), use_bias=use_bias)
        self.bn2a = tf.keras.layers.BatchNormalization(trainable=train_bn)

        self.conv2b = tf.keras.layers.Conv2D(filters=filters2, kernel_size=self.kernel_size, padding='same',
                                             use_bias=use_bias)
        self.bn2b = tf.keras.layers.BatchNormalization(trainable=train_bn)

        self.conv2c = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), use_bias=use_bias)
        self.bn2c = tf.keras.layers.BatchNormalization(trainable=train_bn)

    def build(self, input_shape):
        self.built = True
        super(ResnetIdentityBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x)

        x += inputs
        return tf.nn.relu(x)

    def get_config(self):
        config = super(ResnetIdentityBlock, self).get_config()
        config.update({'kernel_size': self.kernel_size, 'filters': self.filters})
        return config


@tf.keras.utils.register_keras_serializable()
class ResNetIdentityBlockSmall(tfl.Layer):

    def __init__(self, filter_size, kernel_size, use_bias=True,
                 train_bn=True, name='resnet_identity_block_small', **kwargs):
        super(ResNetIdentityBlockSmall, self).__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.kernel_size = kernel_size

        self.conv2a = tf.keras.layers.Conv2D(filters=self.filter_size, kernel_size=self.kernel_size, use_bias=use_bias)
        self.bn2a = tf.keras.layers.BatchNormalization(trainable=train_bn)

        self.conv2b = tf.keras.layers.Conv2D(filters=self.filter_size, kernel_size=self.kernel_size, use_bias=use_bias)
        self.bn2b = tf.keras.layers.BatchNormalization(trainable=train_bn)

    def build(self, input_shape):
        self.built = True
        super(ResNetIdentityBlockSmall, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)

        x += inputs
        return tf.nn.relu(x)

    def get_config(self):
        config = super(ResNetIdentityBlockSmall, self).get_config()
        config.update({'filter_size': self.filter_size, 'kernel_size': self.kernel_size, })
        return config


@tf.keras.utils.register_keras_serializable()
class ResNetLayer(tfl.Layer):

    def __init__(self, resnet_type, train_bn, name='resnet_layer', stage5=False, **kwargs):
        """
        Build a ResNet graph. https://arxiv.org/pdf/1512.03385.pdf
        Args:
            resnet_type: model type
            name:
            train_bn:
            stage5:
            **kwargs:
        """
        super(ResNetLayer, self).__init__(name=name, **kwargs)
        self.resnet_type = resnet_type
        self.stage5 = stage5
        self.stage4_identity_count = {
            'resnet18': 1,
            'resnet34': 5,
            'resnet50': 5,
            'resnet101': 22}
        assert self.resnet_type in [f'resnet{i}' for i in [18, 34, 50, 101]]

        # Stage 1
        self.zero_pad = tfl.ZeroPadding2D((3, 3))
        self.conv1 = tfl.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', use_bias=True)
        self.bn_conv1 = tfl.BatchNormalization(name='bn_conv1', trainable=train_bn)
        self.maxpool1 = tfl.MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.relu = tfl.Activation('relu')

        self.stages_dict = dict()

        if self.resnet_type in ['resnet50', 'resnet101']:

            # Stage 2
            self.resnet_conv2a = ResnetConvBlock(filters=[64, 64, 256], kernel_size=3, strides=(1, 1),
                                                 train_bn=train_bn,
                                                 name='resnet_conv2a')
            self.resnet_identity2b = ResnetIdentityBlock(filters=[64, 64, 256], kernel_size=3, train_bn=train_bn,
                                                         name='resnet_identity2b')
            self.resnet_identity2c = ResnetIdentityBlock(filters=[64, 64, 256], kernel_size=3, train_bn=train_bn,
                                                         name='resnet_identity2c')

            # Stage 3
            self.resnet_conv3a = ResnetConvBlock(filters=[128, 128, 512], kernel_size=3, train_bn=train_bn,
                                                 name='resnet_conv3a')
            self.resnet_identity3b = ResnetIdentityBlock(filters=[128, 128, 512], kernel_size=3, train_bn=train_bn,
                                                         name='resnet_identity3b')
            self.resnet_identity3c = ResnetIdentityBlock(filters=[128, 128, 512], kernel_size=3, train_bn=train_bn,
                                                         name='resnet_identity3c')
            self.resnet_identity3d = ResnetIdentityBlock(filters=[128, 128, 512], kernel_size=3, train_bn=train_bn,
                                                         name='resnet_identity3d')

            # Stage 4
            self.resnet_conv4a = ResnetConvBlock(filters=[256, 256, 1024], kernel_size=3, train_bn=train_bn)
            self.resnet_identity4_list = []
            for i in range(self.stage4_identity_count[self.resnet_type]):
                self.resnet_identity4_list.append(ResnetIdentityBlock(
                    filters=[256, 256, 1024], kernel_size=3, train_bn=train_bn)
                )
            # Stage 5
            if self.stage5:
                self.resnet_conv5a = ResnetConvBlock(filters=[512, 512, 2048], kernel_size=3, train_bn=train_bn)
                self.resnet_identity5b = ResnetIdentityBlock(filters=[512, 512, 2048], kernel_size=3, train_bn=train_bn)
                self.resnet_identity5c = ResnetIdentityBlock(filters=[512, 512, 2048], kernel_size=3, train_bn=train_bn)

        elif self.resnet_type == 'resnet34':
            # Stage 2
            self.resnet_conv2a = ResnetConvBlockSmall(
                filter_size=64, kernel_size=3, strides=(1, 1), train_bn=True, name='resnet_conv2a')
            self.resnet_identity2b = ResNetIdentityBlockSmall(
                filter_size=64, kernel_size=1, train_bn=True, name='resnet_identity2b')
            self.resnet_identity2c = ResNetIdentityBlockSmall(
                filter_size=64, kernel_size=1, train_bn=True, name='resnet_identity2b')
            # Stage 3
            self.resnet_conv3a = ResnetConvBlockSmall(
                filter_size=128, kernel_size=3, train_bn=True, name='resnet_conv3a')
            self.resnet_identity3b = ResNetIdentityBlockSmall(
                filter_size=128, kernel_size=1, train_bn=True, name='resnet_identity3b')
            self.resnet_identity3c = ResNetIdentityBlockSmall(
                filter_size=128, kernel_size=1, train_bn=True, name='resnet_identity3c')
            self.resnet_identity3d = ResNetIdentityBlockSmall(
                filter_size=128, kernel_size=1, train_bn=True, name='resnet_identity3d')
            # Stage 4
            self.resnet_conv4a = ResnetConvBlockSmall(
                filter_size=256, kernel_size=3, train_bn=True, name='resnet_conv4a')
            self.resnet_identity4_list = []
            for i in range(self.stage4_identity_count[self.resnet_type]):
                self.resnet_identity4_list.append(
                    ResNetIdentityBlockSmall(filter_size=256, kernel_size=1, train_bn=True))
            # Stage 5
            if self.stage5:
                self.resnet_conv5a = ResnetConvBlockSmall(
                    filter_size=512, kernel_size=3, train_bn=True, name='resnet_conv5a')
                self.resnet_identity5b = ResNetIdentityBlockSmall(
                    filter_size=512, kernel_size=1, train_bn=True, name='resnet_identity5b')
                self.resnet_identity5c = ResNetIdentityBlockSmall(
                    filter_size=512, kernel_size=1, train_bn=True, name='resnet_identity5c')

        elif self.resnet_type == 'resnet18':
            # Stage 2
            self.resnet_conv2a = ResnetConvBlockSmall(
                filter_size=64, kernel_size=3, strides=(1, 1), train_bn=True, name='resnet_conv2a')
            self.resnet_identity2b = ResNetIdentityBlockSmall(
                filter_size=64, kernel_size=1, train_bn=True, name='resnet_identity2b')
            # Stage 3
            self.resnet_conv3a = ResnetConvBlockSmall(
                filter_size=128, kernel_size=3, train_bn=True, name='resnet_conv3a')
            self.resnet_identity3b = ResNetIdentityBlockSmall(
                filter_size=128, kernel_size=1, train_bn=True, name='resnet_identity3b')

            # Stage 4
            self.resnet_conv4a = ResnetConvBlockSmall(filter_size=256, kernel_size=3, train_bn=True,
                                                      name='resnet_conv4a')
            self.resnet_identity4b = ResNetIdentityBlockSmall(filter_size=256, kernel_size=1, train_bn=True,
                                                              name='resnet_identity4b')
            # Stage 5
            if self.stage5:
                self.resnet_conv5a = ResnetConvBlockSmall(
                    filter_size=512, kernel_size=3, train_bn=True, name='resnet_conv5a')
                self.resnet_identity5b = ResNetIdentityBlockSmall(
                    filter_size=512, kernel_size=1, train_bn=True, name='resnet_identity5b')

    def build(self, input_shape):
        self.built = True
        super(ResNetLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # outputs comments here are for paper resize 224x224
        features_dict = {}
        # Stage 1
        x = self.zero_pad(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)  # output: (n, 56, 56, 64)
        features_dict.update({'C1': x})

        if self.resnet_type in ['resnet50', 'resnet101']:
            # Stage 2
            x = self.resnet_conv2a(x)
            x = self.resnet_identity2b(x)
            x = self.resnet_identity2c(x)  # output: (n, 56, 56, 256)
            features_dict.update({'C2': x})

            # Stage 3
            x = self.resnet_conv3a(x)
            x = self.resnet_identity3b(x)
            x = self.resnet_identity3c(x)
            x = self.resnet_identity3d(x)  # output: (n, 28, 28, 512)

            features_dict.update({'C3': x})

            # Stage 4
            x = self.resnet_conv4a(x)
            for layer in self.resnet_identity4_list:
                x = layer(x)
            # output: (1, 14, 14, 1024)
            features_dict.update({'C4': x})

            # Stage 5
            features_dict.update({'C5': None})
            if self.stage5:
                x = self.resnet_conv5a(x)
                x = self.resnet_identity5b(x)
                x = self.resnet_identity5c(x)  # output: # (n, 7, 7, 2048)
                features_dict.update({'C5': x})

        elif self.resnet_type == 'resnet34':
            # Stage 2
            x = self.resnet_conv2a(x)
            x = self.resnet_identity2b(x)
            x = self.resnet_identity2c(x)
            features_dict.update({'C2': x})  # output: (n, 56, 56, 64)

            # Stage 3
            x = self.resnet_conv3a(x)
            x = self.resnet_identity3b(x)
            x = self.resnet_identity3c(x)
            x = self.resnet_identity3d(x)
            features_dict.update({'C3': x})  # output: (n, 28, 28, 128)

            # Stage 4
            x = self.resnet_conv4a(x)
            for layer in self.resnet_identity4_list:
                x = layer(x)
            features_dict.update({'C4': x})  # output: (n, 14, 14, 256)

            # Stage 5
            features_dict.update({'C5': None})
            if self.stage5:
                x = self.resnet_conv5a(x)
                x = self.resnet_identity5b(x)
                x = self.resnet_identity5c(x)
                features_dict.update({'C5': x})  # output: (n, 7, 7, 512)

        elif self.resnet_type == 'resnet18':
            # Stage 2
            x = self.resnet_conv2a(x)
            x = self.resnet_identity2b(x)
            features_dict.update({'C2': x})  # output: (n, 56, 56, 64)

            # Stage 3
            x = self.resnet_conv3a(x)
            x = self.resnet_identity3b(x)
            features_dict.update({'C3': x})  # output: (n, 28, 28, 128)

            # Stage 4
            x = self.resnet_conv4a(x)
            x = self.resnet_identity4b(x)
            features_dict.update({'C4': x})  # output: (n, 14, 14, 256)

            # Stage 5
            features_dict.update({'C5': None})
            if self.stage5:
                x = self.resnet_conv5a(x)
                x = self.resnet_identity5b(x)
                features_dict.update({'C5': x})  # output: (n, 7, 7, 512)

        return features_dict['C2'], features_dict['C3'], features_dict['C4'], features_dict['C5']

    def get_config(self):
        config = super(ResNetLayer, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class UpSamplingLayer(tfl.Layer):

    def __init__(self, config, name='upsampling_layer', **kwargs):
        self.config = config
        super(UpSamplingLayer, self).__init__(name=name, **kwargs)

        self.p5_conv = tfl.Conv2D(self.config['top_down_pyramid_size'], (1, 1), name='fpn_c5p5')

        self.p5_upsample = tfl.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")
        self.p4_conv = tfl.Conv2D(self.config['top_down_pyramid_size'], (1, 1), name='fpn_c4p4')

        self.p4_upsample = tfl.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")
        self.p3_conv = tfl.Conv2D(self.config['top_down_pyramid_size'], (1, 1), name='fpn_c3p3')

        self.p3_upsample = tfl.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")
        self.p2_conv = tfl.Conv2D(self.config['top_down_pyramid_size'], (1, 1), name='fpn_c2p2')

        self.fpn_p2 = tfl.Conv2D(self.config['top_down_pyramid_size'], (3, 3), padding="same", name="fpn_p2")
        self.fpn_p3 = tfl.Conv2D(self.config['top_down_pyramid_size'], (3, 3), padding="same", name="fpn_p3")
        self.fpn_p4 = tfl.Conv2D(self.config['top_down_pyramid_size'], (3, 3), padding="same", name="fpn_p4")
        self.fpn_p5 = tfl.Conv2D(self.config['top_down_pyramid_size'], (3, 3), padding="same", name="fpn_p5")
        self.fpn_p6 = tfl.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")

    def build(self, input_shape):
        self.built = True
        super(UpSamplingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # c2 shape=(n, 128, 128, filters),
        # c3 shape=(n, 64, 64,   filters),
        # c4 shape=(n, 32, 32,   filters),
        # c5 shape=(n, 16, 16,   filters)
        c2, c3, c4, c5 = inputs
        # Top-down Layers
        p5 = self.p5_conv(c5)  # (n ,16,16,256)
        p4 = self.p5_upsample(p5) + self.p4_conv(c4)  # (n, 32, 32, filters)
        p3 = self.p4_upsample(p4) + self.p3_conv(c3)  # (n, 64, 64, filters)
        p2 = self.p3_upsample(p3) + self.p2_conv(c2)  # (n, 128, 128, filters)
        # Attach 3x3 conv to all p layers to get the final feature maps.
        p2 = self.fpn_p2(p2)  # (n, 128, 128, filters)
        p3 = self.fpn_p3(p3)  # (n, 64, 64, filters)
        p4 = self.fpn_p4(p4)  # (n, 32, 32, filters)
        p5 = self.fpn_p5(p5)  # (n ,16,16,filters)
        # p6 is used for the 5th anchor scale in RPN. Generated by subsampling from p5 with stride of 2.
        p6 = self.fpn_p6(p5)  # (n, 8, 8, filters)

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        return rpn_feature_maps, mrcnn_feature_maps

    def get_config(self):
        config = super(UpSamplingLayer, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class AnchorsLayer(tfl.Layer):
    def __init__(self, config, training, name="anchors", **kwargs):
        super(AnchorsLayer, self).__init__(name=name, **kwargs)
        self.config = config
        self.training = training
        self.image_shape = self.config['image_shape']
        self.norm_boxes_layer = NormBoxesLayer(name='norm_boxes_anchors')

        self.anchors = self.get_anchors()
        self.anchors = tf.Variable(self.anchors, trainable=False)

    def get_anchors(self):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = utils.compute_backbone_shapes(self.config)
        # [128 128] [64..] [32..] [16..] [8..]
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(self.image_shape) in self._anchor_cache:
            # Generate Anchors
            self.anchors = utils.generate_pyramid_anchors(
                self.config['rpn_anchor_scales'],
                self.config['rpn_anchor_ratios'],
                backbone_shapes,
                self.config['backbone_strides'],
                self.config['rpn_anchor_stride'])
            # Normalize coordinates
            self._anchor_cache[tuple(self.image_shape)] = self.norm_boxes_layer([self.anchors, self.image_shape[:2]])
        anchors = self._anchor_cache[tuple(self.image_shape)]
        return np.broadcast_to(anchors, (self.config['batch_size'],) + anchors.shape)

    def build(self, input_shape):
        self.built = True
        super(AnchorsLayer, self).build(input_shape)

    def call(self, dummy=None, **kwargs):
        return self.anchors

    def get_config(self):
        config = super(AnchorsLayer, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class RPNLayer(tfl.Layer):

    def __init__(self, anchors_per_location, anchor_stride, name='rpn_layer', **kwargs):
        super(RPNLayer, self).__init__(name=name, **kwargs)
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride

        self.conv_shared = tfl.Conv2D(512, (3, 3), padding='same', activation=None,
                                      strides=anchor_stride, name='rpn_conv_shared',
                                      )

        self.relu = tfl.Activation('relu')
        self.rpn_class_raw = tfl.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                                        activation='linear', name='rpn_class_raw',
                                        )
        self.rpn_class_xxx = tfl.Activation("softmax", name="rpn_class_xxx")

        self.rpn_bbox_pred = tfl.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                                        activation='linear', name='rpn_bbox_pred',
                                        )

        self.reshape_logits = tfl.Reshape((-1, 2),
                                          name='reshape_logits')  # tfl.Lambda(lambda x: tf.reshape(x, (-1, 2)), name='reshape_logits')
        self.reshape_rpn_box = tfl.Reshape((-1, 4),
                                           name='reshape_rpn_box')  # tfl.Lambda(lambda x: tf.reshape(x, (-1, 4)), name='reshape_rpn_box')

    def build(self, input_shape):
        self.built = True
        super(RPNLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: a list of rpn_feature_maps = [p2, p3, p4, p5, p6] from UpSamplingLayer
            **kwargs:

        Returns: tuple (rpn_class_logits, rpn_probs, rpn_bbox)
                 with dimensions (1, 49152, 2) (1, 49152, 2) (1, 49152, 4)

        """
        # Shared convolutional base of the RPN
        shared_x = self.relu(self.conv_shared(inputs))
        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = self.rpn_class_raw(shared_x)
        # Reshape to [batch, anchors, 2]
        rpn_class_logits = self.reshape_logits(x)
        rpn_probs = self.rpn_class_xxx(rpn_class_logits)
        x = self.rpn_bbox_pred(shared_x)
        rpn_bbox = self.reshape_rpn_box(x)

        return rpn_class_logits, rpn_probs, rpn_bbox

    def get_config(self):
        config = super(RPNLayer, self).get_config()
        config.update({"anchors_per_location": self.anchors_per_location,
                       "anchor_stride": self.anchor_stride
                       })
        return config


@tf.keras.utils.register_keras_serializable()
class ProposalLayer(tfl.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Args:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, config, name='roi', **kwargs):
        super(ProposalLayer, self).__init__(name=name, **kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = self.config['rpn_nms_threshold']

    def nms(self, boxes, scores):
        indices = tf.image.non_max_suppression(
            boxes, scores, self.proposal_count,
            self.nms_threshold, name="rpn_non_max_suppression")
        proposals = tf.gather(boxes, indices)
        # Pad if needed
        padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals

    def call(self, inputs, **kwargs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config['rpn_bbox_std_dev'], [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.

        pre_nms_limit = tf.minimum(self.config['pre_nms_limit'], tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config['images_per_gpu'])
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config['images_per_gpu'])
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                            self.config['images_per_gpu'],
                                            names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: utils.apply_box_deltas_graph(x, y),
                                  self.config['images_per_gpu'],
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: utils.clip_boxes_graph(x, window),
                                  self.config['images_per_gpu'],
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        proposals = utils.batch_slice([boxes, scores], self.nms, self.config['images_per_gpu'])
        return proposals

    def build(self, input_shape):
        self.built = True
        super(ProposalLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return None, self.proposal_count, 4

    def get_config(self):
        config = super(ProposalLayer, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class DetectionTargetLayer(tfl.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, name='proposal_targets', **kwargs):
        super(DetectionTargetLayer, self).__init__(name=name, **kwargs)
        self.config = config

    def call(self, inputs, **kwargs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        names = ["rois", "target_class_ids", "target_bbox_deltas", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config['images_per_gpu'], names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config['train_rois_per_image'], 4),  # rois
            (None, self.config['train_rois_per_image']),  # class_ids
            (None, self.config['train_rois_per_image'], 4),  # deltas
            (None, self.config['train_rois_per_image'], self.config['mask_shape'][0], self.config['mask_shape'][1])
            # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]

    def get_config(self):
        config = super(DetectionTargetLayer, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class DetectionLayer(tfl.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, proposals, detection_min_confidence, detection_max_instances, detection_nms_threshold,
                 bbox_std_dev, images_per_gpu, batch_size, name='mrcnn_detection', **kwargs):
        super(DetectionLayer, self).__init__(name=name, **kwargs)
        self.detection_min_confidence = detection_min_confidence
        self.detection_max_instances = detection_max_instances
        self.detection_nms_threshold = detection_nms_threshold
        self.bbox_std_dev = bbox_std_dev
        self.batch_size = batch_size
        self.proposals = proposals
        self.images_per_gpu = images_per_gpu
        self.norm_boxes_layer = NormBoxesLayer(name='norm_boxes_detection')

    def build(self, input_shape):
        self.built = True
        super(DetectionLayer, self).build(input_shape)

    def refine_detections(self, rois, probs, deltas, window):
        """Refine classified proposals and filter overlaps and return final
         detections.

         Inputs:
             rois: [N, (y1, x1, y2, x2)] in normalized coordinates
             probs: [N, num_classes]. Class probabilities.
             deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                     bounding box deltas.
             window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
                 that contains the image excluding the padding.

         Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
             coordinates are normalized.
         """
        # Class IDs per ROI
        class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
        # Class probability of the top class of each ROI
        # indices = tf.stack([tf.range(self.proposals), class_ids], axis=1)
        proposals_range = tf.range(self.proposals)
        indices = tf.concat([tf.reshape(proposals_range, (tf.shape(proposals_range)[0], 1)),
                             tf.reshape(class_ids, (tf.shape(class_ids)[0], 1))],
                            axis=1)
        class_scores = tf.gather_nd(probs, indices)
        # Class-specific bounding box deltas
        deltas_specific = tf.gather_nd(deltas, indices)
        # Apply bounding box deltas
        # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
        refined_rois = utils.apply_box_deltas_graph(
            rois, tf.math.multiply(deltas_specific, self.bbox_std_dev))
        # Clip boxes to image window
        refined_rois = utils.clip_boxes_graph(refined_rois, window)

        # TODO: Filter out boxes with zero area
        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        # Filter out low confidence boxes
        if self.detection_min_confidence:
            conf_keep = tf.where(class_scores >= self.detection_min_confidence)[:, 0]

            """
            keep = tf.sets.intersection(tf.expand_dims(keep, 0),tf.expand_dims(conf_keep, 0))
            keep = tf.sparse.to_dense(keep)[0]
            """
            broadcast_equal = tf.equal(conf_keep, tf.reshape(keep, (-1, 1)))
            broadcast_equal_int = tf.cast(broadcast_equal, tf.int32)
            broadcast_sum = tf.reduce_sum(broadcast_equal_int, axis=0)
            keep = tf.boolean_mask(conf_keep, broadcast_sum, axis=None)

        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois, keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            """Apply Non-Maximum Suppression on ROIs of the given class."""
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=self.detection_max_instances,
                iou_threshold=self.detection_nms_threshold)
            # Map indices
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            # Pad with -1 so returned tensors have the same shape
            gap = self.detection_max_instances - tf.shape(class_keep)[0]
            class_keep = tf.pad(class_keep, [(0, gap)],
                                mode='CONSTANT', constant_values=-1)
            # Set shape so map_fn() can infer result shape
            # class_keep.set_shape([self.detection_max_instances])
            return class_keep

        @tf.function
        def _nms_keep_func(class_ids):
            """
            An experimental function for replacing nms_keep_map with map_fn.
            TODO: check/fix for multiple classes
            Args:
                class_ids:
            Returns: class_keep

            """
            broadcast_equal = tf.equal(pre_nms_class_ids, tf.reshape(class_ids, (-1, 1)))
            broadcast_equal_int = tf.cast(broadcast_equal, tf.int32)
            bool_mask = tf.reduce_sum(broadcast_equal_int, axis=0)

            # Apply NMS
            class_keep = tf.image.non_max_suppression(tf.boolean_mask(
                pre_nms_rois, bool_mask, axis=None),
                tf.boolean_mask(pre_nms_scores, bool_mask, axis=None),
                max_output_size=self.detection_max_instances,
                iou_threshold=self.detection_nms_threshold)
            # Map indicies
            class_keep = tf.gather(keep, class_keep)
            # Pad with -1 so returned tensors have the same shape
            gap = self.detection_max_instances - tf.shape(class_keep)[0]
            class_keep = tf.pad(class_keep, [(0, gap)],
                                mode='CONSTANT', constant_values=-1)
            return class_keep

        # 2. Map over class IDs
        # nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
        nms_keep = _nms_keep_func(unique_pre_nms_class_ids)

        # 3. Merge results into one list, and remove -1 padding
        # nms_keep = tf.reshape(nms_keep, [-1])
        nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
        # 4. Compute intersection between keep and nms_keep

        broadcast_equal = tf.equal(nms_keep, tf.reshape(keep, (-1, 1)))
        broadcast_equal_int = tf.cast(broadcast_equal, tf.int32)
        broadcast_sum = tf.reduce_sum(broadcast_equal_int, axis=0)
        keep = tf.boolean_mask(nms_keep, broadcast_sum, axis=None)
        """
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]
        """

        # Keep top detections
        roi_count = self.detection_max_instances
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are normalized.
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), dtype='float32')[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

        # Pad with zeros if detections < DETECTION_MAX_INSTANCES
        gap = self.detection_max_instances - tf.shape(detections)[0]
        detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
        return detections

    def call(self, inputs, **kwargs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = utils.parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = self.norm_boxes_layer([m['window'], image_shape[:2]])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: self.refine_detections(x, y, w, z),
            self.images_per_gpu)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.batch_size, self.detection_max_instances, 6])

    def compute_output_shape(self, input_shape):
        return None, self.detection_max_instances, 6

    def get_config(self):
        config = super(DetectionLayer, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class DetectedBoxesExtraction(tfl.Layer):

    def __init__(self, config=None, name='detected_boxes_extraction', **kwargs):
        super(DetectedBoxesExtraction, self).__init__(name=name, **kwargs)
        self.config = config

    def build(self, input_shape):
        self.built = True
        super(DetectedBoxesExtraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs[..., :4]

    def get_config(self):
        config = super(DetectedBoxesExtraction, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class PyramidROIAlign(tfl.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, name='roi_align', **kwargs):
        super(PyramidROIAlign, self).__init__(name=name, **kwargs)
        self.pool_shape = tuple(pool_shape)

    def build(self, input_shape):
        self.built = True
        super(PyramidROIAlign, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = utils.parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = utils.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        # Workaround for onnxruntime which have issues with empty tensors concat because of the dimensions reorder.
        unique_levels = tf.unique(tf.reshape(roi_level, (-1, )))[0]
        unique_levels_padded = tf.pad(unique_levels, tf.constant([[0, 4]]), constant_values=2)
        unique_levels_padded = tf.split(unique_levels_padded[:4], 4)

        for i, level in enumerate(unique_levels_padded):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)[:tf.shape(boxes)[0]*tf.shape(boxes)[1]]

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)[:tf.shape(boxes)[0]*tf.shape(boxes)[1]]

        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)

    def get_config(self):
        config = super(PyramidROIAlign, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class FPNClassifier(tfl.Layer):

    def __init__(self, pool_size, fc_layers_size, num_classes, train_bn, name='fpn_classifier', **kwargs):
        super(FPNClassifier, self).__init__(name=name, **kwargs)
        self.pool_size = pool_size
        self.fc_layers_size = fc_layers_size
        self.num_classes = num_classes
        # ROI Pooling
        # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        self.roi_align = PyramidROIAlign([self.pool_size, self.pool_size], name="roi_align_classifier")
        # Two 1024 FC layers (implemented with Conv2D for consistency)

        self.fpnclf_conv1 = tfl.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid", name='fpnclf_conv1')
        self.tdistr_conv1 = tfl.TimeDistributed(self.fpnclf_conv1, name="mrcnn_class_conv1")

        self.fpnclf_bn1 = tfl.BatchNormalization(trainable=train_bn, name='fpnclf_bn1')
        self.tdistr_bn1 = tfl.TimeDistributed(self.fpnclf_bn1, name='mrcnn_class_bn1')
        self.fpnclf_relu1 = tfl.Activation('relu', name='fpnclf_relu1')

        self.fpnclf_conv2 = tfl.Conv2D(fc_layers_size, (1, 1), name='fpnclf_conv2')
        self.tdistr_conv2 = tfl.TimeDistributed(self.fpnclf_conv2, name="mrcnn_class_conv2")

        self.fpnclf_bn2 = tfl.BatchNormalization(trainable=train_bn, name='fpnclf_bn2')
        self.tdistr_bn2 = tfl.TimeDistributed(self.fpnclf_bn2, name='mrcnn_class_bn2')
        self.fpnclf_relu2 = tfl.Activation('relu', name='fpnclf_relu2')

        # Classifier head
        self.mrcnn_class_logits_layer = tfl.TimeDistributed(tfl.Dense(num_classes), name='mrcnn_class_logits')
        self.mrcnn_probs_layer = tfl.TimeDistributed(tfl.Activation("softmax"), name="mrcnn_class")

        # BBox head
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        self.mrcnn_bbox_fc = tfl.TimeDistributed(tfl.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')

        self.pool_squeeze = tfl.Lambda(lambda x: tf.squeeze(tf.squeeze(x, 3), 2), name='fpn_clf_pool_squeeze')

    def build(self, input_shape):
        self.built = True
        super(FPNClassifier, self).build(input_shape)

    def call(self, inputs, **kwargs):
        rois, image_meta, feature_maps = inputs

        x = self.roi_align([rois, image_meta] + feature_maps)
        x = self.tdistr_bn1(self.tdistr_conv1(x))
        x = self.fpnclf_relu1(x)
        x = self.tdistr_bn2(self.tdistr_conv2(x))
        x = self.fpnclf_relu2(x)

        shared = self.pool_squeeze(x)
        mrcnn_class_logits = self.mrcnn_class_logits_layer(shared)
        mrcnn_probs = self.mrcnn_probs_layer(mrcnn_class_logits)

        x = self.mrcnn_bbox_fc(shared)
        # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
        mrcnn_bbox = tf.reshape(
            tensor=x,
            shape=(tf.shape(x)[0], tf.shape(x)[1], self.num_classes, 4),
            name="mrcnn_bbox")

        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

    def get_config(self):
        config = super(FPNClassifier, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class FPNMaskLayer(tfl.Layer):

    def __init__(self, pool_size, num_classes, train_bn, name='fpn_mask_layer', **kwargs):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        """
        super(FPNMaskLayer, self).__init__(name=name, **kwargs)

        self.roi_align = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")

        self.fpnmask_conv1 = tfl.Conv2D(256, (3, 3), padding="same", name='fpnmask_conv1')
        self.tdistr_conv1 = tfl.TimeDistributed(self.fpnmask_conv1, name="mrcnn_mask_conv1")
        self.fpnmask_bn1 = tfl.BatchNormalization(trainable=train_bn, name='fpnmask_bn1')
        self.tdistr_bn1 = tfl.TimeDistributed(self.fpnmask_bn1, name='mrcnn_mask_bn1')

        self.fpnmask_conv2 = tfl.Conv2D(256, (3, 3), padding="same", name='fpnmask_conv2')
        self.tdistr_conv2 = tfl.TimeDistributed(self.fpnmask_conv2, name="mrcnn_mask_conv2")
        self.fpnmask_bn2 = tfl.BatchNormalization(trainable=train_bn, name='fpnmask_bn2')
        self.tdistr_bn2 = tfl.TimeDistributed(self.fpnmask_bn2, name='mrcnn_mask_bn2')

        self.fpnmask_conv3 = tfl.Conv2D(256, (3, 3), padding="same", name='fpnmask_conv3')
        self.tdistr_conv3 = tfl.TimeDistributed(self.fpnmask_conv3, name="mrcnn_mask_conv3")
        self.fpnmask_bn3 = tfl.BatchNormalization(trainable=train_bn, name='fpnmask_bn3')
        self.tdistr_bn3 = tfl.TimeDistributed(self.fpnmask_bn3, name='mrcnn_mask_bn3')

        self.fpnmask_conv4 = tfl.Conv2D(256, (3, 3), padding="same", name='fpnmask_conv4')
        self.tdistr_conv4 = tfl.TimeDistributed(self.fpnmask_conv4, name="mrcnn_mask_conv4")
        self.fpnmask_bn4 = tfl.BatchNormalization(trainable=train_bn, name='fpnmask_bn4')
        self.tdistr_bn4 = tfl.TimeDistributed(self.fpnmask_bn4, name='mrcnn_mask_bn4')

        self.fpnmask_deconv = tfl.Conv2DTranspose(256, (2, 2), strides=2, activation="relu", name='fpnmask_convt')
        self.tdistr_deconv = tfl.TimeDistributed(self.fpnmask_deconv, name="mrcnn_mask_deconv")

        self.fpn_conv_ = tfl.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid", name='fpn_conv_')
        self.tdistr_mrcnn_mask = tfl.TimeDistributed(self.fpn_conv_, name="mrcnn_mask")

        self.relu = tfl.Activation('relu')

    def build(self, input_shape):
        self.built = True
        super(FPNMaskLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        rois, image_meta, feature_maps = inputs

        x = self.roi_align([rois, image_meta] + feature_maps)
        x = self.tdistr_bn1(self.tdistr_conv1(x))
        x = self.relu(x)

        x = self.tdistr_bn2(self.tdistr_conv2(x))
        x = self.relu(x)

        x = self.tdistr_bn3(self.tdistr_conv3(x))
        x = self.relu(x)

        x = self.tdistr_bn4(self.tdistr_conv4(x))
        x = self.relu(x)

        x = self.tdistr_deconv(x)
        x = self.tdistr_mrcnn_mask(x)

        return x

    def get_config(self):
        config = super(FPNMaskLayer, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class ImageMetaLayer(tfl.Layer):
    def __init__(self, name='parse_image_meta_layer', **kwargs):
        super(ImageMetaLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.built = True
        super(ImageMetaLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        image_id = inputs[:, 0]
        original_image_shape = inputs[:, 1:4]
        image_shape = inputs[:, 4:7]
        window = inputs[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
        scale = inputs[:, 11]
        active_class_ids = inputs[:, 12:]
        total_dict = {"image_id": image_id,
                      "original_image_shape": original_image_shape,
                      "image_shape": image_shape,
                      "window": window,
                      "scale": scale,
                      "active_class_ids": active_class_ids,
                      }
        return total_dict


# Functional tf.keras API
def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.math.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.math.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config['train_rois_per_image'] *
                         config['roi_positive_ratio'])
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config['roi_positive_ratio']
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config['bbox_std_dev']

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config['use_mini_masks']:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config['mask_shape'])
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config['train_rois_per_image'] - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.math.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.math.maximum(b1_y1, b2_y1)
    x1 = tf.math.maximum(b1_x1, b2_x1)
    y2 = tf.math.minimum(b1_y2, b2_y2)
    x2 = tf.math.minimum(b1_x2, b2_x2)
    intersection = tf.math.maximum(x2 - x1, 0) * tf.math.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'bbone_res' + str(stage) + block + '_branch'
    bn_name_base = 'bbone_bn' + str(stage) + block + '_branch'

    x = tfl.Conv2D(nb_filter1, (1, 1), strides=strides,
                   name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = tfl.BatchNormalization(name=bn_name_base + '2a', trainable=train_bn)(x)
    x = tfl.Activation('relu', name=f'relu_2a_{block}_{stage}')(x)

    x = tfl.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                   name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = tfl.BatchNormalization(name=bn_name_base + '2b', trainable=train_bn)(x)
    x = tfl.Activation('relu', name=f'relu_2b_{block}_{stage}')(x)

    x = tfl.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                            '2c', use_bias=use_bias)(x)
    x = tfl.BatchNormalization(name=bn_name_base + '2c', trainable=train_bn)(x)

    shortcut = tfl.Conv2D(nb_filter3, (1, 1), strides=strides,
                          name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = tfl.BatchNormalization(name=bn_name_base + '1', trainable=train_bn)(shortcut)

    x = tfl.Add(name=f'add_{block}_{stage}')([x, shortcut])
    x = tfl.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block_small(input_tensor, filter_size, stage, block, kernel_size=3, strides=(2, 2),
                     use_bias=True, train_bn=True):
    conv_name_base = 'bbone_res' + str(stage) + block + '_branch'
    bn_name_base = 'bbone_bn' + str(stage) + block + '_branch'

    x = tfl.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides,
                   use_bias=use_bias, padding='same', name=conv_name_base + '2a')(input_tensor)
    x = tfl.BatchNormalization(trainable=train_bn, name=bn_name_base + '2a')(x)
    x = tfl.Activation('relu', name=f'relu_2a_{block}_{stage}')(x)

    x = tfl.Conv2D(filters=filter_size, kernel_size=kernel_size, use_bias=use_bias,
                   padding='same', name=conv_name_base + '2b')(x)
    x = tfl.BatchNormalization(trainable=train_bn, name=bn_name_base + '2b')(x)
    x = tfl.Activation('relu', name=f'relu_2b_{block}_{stage}')(x)

    shortcut = tfl.Conv2D(filters=filter_size, kernel_size=(1, 1), strides=strides,
                          use_bias=use_bias, name=conv_name_base + '1')(input_tensor)
    shortcut = tfl.BatchNormalization(trainable=train_bn, name=bn_name_base + '1')(shortcut)

    x = tfl.Add(name=f'add_{block}_{stage}')([x, shortcut])
    x = tfl.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def identity_block_small(input_tensor, filter_size, stage, block, kernel_size,
                         use_bias=True, train_bn=True):
    conv_name_base = 'bbone_res' + str(stage) + block + '_branch'
    bn_name_base = 'bbone_bn' + str(stage) + block + '_branch'

    x = tfl.Conv2D(filters=filter_size, kernel_size=kernel_size, use_bias=use_bias,
                   name=conv_name_base + '2a')(input_tensor)
    x = tfl.BatchNormalization(name=bn_name_base + '2a', trainable=train_bn)(x)
    x = tfl.Activation('relu', name=f'relu_2a_{block}_{stage}')(x)

    x = tfl.Conv2D(filters=filter_size, kernel_size=kernel_size, use_bias=use_bias,
                   name=conv_name_base + '2b')(x)
    x = tfl.BatchNormalization(name=bn_name_base + '2b', trainable=train_bn)(x)

    x = tfl.Add(name=f'add_{block}_{stage}')([x, input_tensor])
    x = tfl.Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """
    The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'bbone_res' + str(stage) + block + '_branch'
    bn_name_base = 'bbone_bn' + str(stage) + block + '_branch'

    x = tfl.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                   use_bias=use_bias)(input_tensor)
    x = tfl.BatchNormalization(name=bn_name_base + '2a', trainable=train_bn)(x)
    x = tfl.Activation('relu', name=f'relu_2a_{block}_{stage}')(x)

    x = tfl.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                   name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = tfl.BatchNormalization(name=bn_name_base + '2b', trainable=train_bn)(x)
    x = tfl.Activation('relu', name=f'relu_2b_{block}_{stage}')(x)

    x = tfl.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                   use_bias=use_bias)(x)
    x = tfl.BatchNormalization(name=bn_name_base + '2c', trainable=train_bn)(x)

    x = tfl.Add(name=f'add_{block}_{stage}')([x, input_tensor])
    x = tfl.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: resnet type
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet18", "resnet34", "resnet50", "resnet101"]
    block_count = {"resnet18": 1, "resnet34": 5, "resnet50": 5, "resnet101": 22}[architecture]

    # Stage 1
    x = tfl.ZeroPadding2D((3, 3))(input_image)
    x = tfl.Conv2D(64, (7, 7), strides=(2, 2), name='bbone_res_conv1', use_bias=True)(x)
    x = tfl.BatchNormalization(trainable=train_bn, name='bbone_bn_conv1')(x)
    x = tfl.Activation('relu')(x)
    C1 = x = tfl.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    if int(architecture.replace('resnet', '')) in [50, 101]:
        # Stage 2
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
        C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
        # Stage 3
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
        C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
        # Stage 4
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)

        for i in range(block_count):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
        C4 = x
        # Stage 5
        if stage5:
            x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
            C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
        else:
            C5 = None

    elif int(architecture.replace('resnet', '')) == 34:
        # Stage 2
        x = conv_block_small(x, filter_size=64, stage=2, block='a', kernel_size=3, strides=(1, 1), train_bn=train_bn)
        x = identity_block_small(x, filter_size=64, stage=2, block='b', kernel_size=1, train_bn=train_bn)
        C2 = x = identity_block_small(x, filter_size=64, stage=2, block='c', kernel_size=1, train_bn=train_bn)

        # Stage 3
        x = conv_block_small(x, filter_size=128, stage=3, block='a', kernel_size=3, train_bn=train_bn)
        x = identity_block_small(x, filter_size=128, stage=3, block='b', kernel_size=1, train_bn=train_bn)
        x = identity_block_small(x, filter_size=128, stage=3, block='c', kernel_size=1, train_bn=train_bn)
        C3 = x = identity_block_small(x, filter_size=128, stage=3, block='d', kernel_size=1, train_bn=train_bn)

        # Stage 4
        x = conv_block_small(x, filter_size=256, stage=4, block='a', kernel_size=3, train_bn=train_bn)
        for i in range(block_count):
            x = identity_block_small(x, filter_size=256, stage=4, block=chr(98 + i), kernel_size=1, train_bn=train_bn)
        C4 = x

        # Stage 5
        if stage5:
            x = conv_block_small(x, filter_size=512, stage=5, block='a', kernel_size=3, train_bn=train_bn)
            x = identity_block_small(x, filter_size=512, stage=5, block='b', kernel_size=1, train_bn=train_bn)
            C5 = x = identity_block_small(x, filter_size=512, stage=5, block='c', kernel_size=1, train_bn=train_bn)
        else:
            C5 = None

    elif int(architecture.replace('resnet', '')) == 18:
        # Stage 2
        x = conv_block_small(x, filter_size=64, stage=2, block='a', kernel_size=3, strides=(1, 1), train_bn=train_bn)
        C2 = x = identity_block_small(x, filter_size=64, stage=2, block='b', kernel_size=1, train_bn=train_bn)

        # Stage 3
        x = conv_block_small(x, filter_size=128, stage=3, block='a', kernel_size=3, train_bn=train_bn)
        C3 = x = identity_block_small(x, filter_size=128, stage=3, block='b', kernel_size=1, train_bn=train_bn)

        # Stage 4
        x = conv_block_small(x, filter_size=256, stage=4, block='a', kernel_size=3, train_bn=train_bn)
        for i in range(block_count):
            x = identity_block_small(x, filter_size=256, stage=4, block=chr(98 + i), kernel_size=1, train_bn=train_bn)
        C4 = x

        # Stage 5
        if stage5:
            x = conv_block_small(x, filter_size=512, stage=5, block='a', kernel_size=3, train_bn=train_bn)
            C5 = x = identity_block_small(x, filter_size=512, stage=5, block='b', kernel_size=1, train_bn=train_bn)
        else:
            C5 = None

    return [C1, C2, C3, C4, C5]


def upsampling_graph(inputs, config):
    p5_conv = tfl.Conv2D(config['top_down_pyramid_size'], (1, 1), name='fpn_c5p5')

    p5_upsample = tfl.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")
    p4_conv = tfl.Conv2D(config['top_down_pyramid_size'], (1, 1), name='fpn_c4p4')

    p4_upsample = tfl.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")
    p3_conv = tfl.Conv2D(config['top_down_pyramid_size'], (1, 1), name='fpn_c3p3')

    p3_upsample = tfl.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")
    p2_conv = tfl.Conv2D(config['top_down_pyramid_size'], (1, 1), name='fpn_c2p2')

    fpn_p2 = tfl.Conv2D(config['top_down_pyramid_size'], (3, 3), padding="same", name="fpn_p2")
    fpn_p3 = tfl.Conv2D(config['top_down_pyramid_size'], (3, 3), padding="same", name="fpn_p3")
    fpn_p4 = tfl.Conv2D(config['top_down_pyramid_size'], (3, 3), padding="same", name="fpn_p4")
    fpn_p5 = tfl.Conv2D(config['top_down_pyramid_size'], (3, 3), padding="same", name="fpn_p5")
    fpn_p6 = tfl.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")

    # c2 shape=(n, 128, 128, filters),
    # c3 shape=(n, 64, 64,   filters),
    # c4 shape=(n, 32, 32,   filters),
    # c5 shape=(n, 16, 16,   filters)
    c2, c3, c4, c5 = inputs
    # Top-down Layers
    p5 = p5_conv(c5)  # (n ,16,16,256)
    p4 = p5_upsample(p5) + p4_conv(c4)  # (n, 32, 32, filters)
    p3 = p4_upsample(p4) + p3_conv(c3)  # (n, 64, 64, filters)
    p2 = p3_upsample(p3) + p2_conv(c2)  # (n, 128, 128, filters)
    # Attach 3x3 conv to all p layers to get the final feature maps.
    p2 = fpn_p2(p2)  # (n, 128, 128, filters)
    p3 = fpn_p3(p3)  # (n, 64, 64, filters)
    p4 = fpn_p4(p4)  # (n, 32, 32, filters)
    p5 = fpn_p5(p5)  # (n ,16,16,filters)
    # p6 is used for the 5th anchor scale in RPN. Generated by subsampling from p5 with stride of 2.
    p6 = fpn_p6(p5)  # (n, 8, 8, filters)

    rpn_feature_maps = [p2, p3, p4, p5, p6]
    mrcnn_feature_maps = [p2, p3, p4, p5]

    return rpn_feature_maps, mrcnn_feature_maps


def rpn_graph(inputs, anchors_per_location, anchor_stride, training):
    """
    use_bias=False for onnx convertion. There is a problem with biases and loop.
    Args:
        inputs:
        anchors_per_location:
        anchor_stride:
        training: bool value

    Returns:

    """
    shared = tfl.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride, use_bias=False,
                        name='rpn_conv_shared')(inputs)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    rpn_class_x = tfl.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', use_bias=False,
                             name='rpn_class_raw')(shared)
    rpn_class_x = tfl.Activation('linear')(rpn_class_x)

    # Reshape to [batch, anchors, 2]
    if training:
        rpn_class_logits = tfl.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(rpn_class_x)
    else:
        rpn_class_logits = tfl.Reshape(target_shape=(-1, 2), name='rpn_class_logits_reshape')(rpn_class_x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = tfl.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    rpn_bbox_x = tfl.Conv2D(anchors_per_location * 4, (1, 1), padding="valid", use_bias=False,
                            name='rpn_bbox_pred')(shared)
    rpn_bbox_x = tfl.Activation('linear')(rpn_bbox_x)

    # Reshape to [batch, anchors, 4]
    if training:
        rpn_bbox = tfl.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]), name='rpn_bbox_reshape')(rpn_bbox_x)
    else:
        rpn_bbox = tfl.Reshape(target_shape=(-1, 4), name='rpn_bbox_reshape')(rpn_bbox_x)

    return rpn_class_logits, rpn_probs, rpn_bbox


def build_rpn_model(anchor_stride, anchors_per_location, depth, training):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = tfl.Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride, training)
    return tf.keras.Model(inputs=[input_feature_map], outputs=outputs, name="rpn_model")


def fpn_classifier_graph(inputs, pool_size, fc_layers_size, num_classes, train_bn,
                         batch_size, post_nms_rois_inference, training):
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    roi_align = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")
    # Two 1024 FC layers (implemented with Conv2D for consistency)

    tdistr_conv1 = tfl.TimeDistributed(tfl.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                                       name="mrcnn_class_conv1")
    tdistr_bn1 = tfl.TimeDistributed(tfl.BatchNormalization(trainable=train_bn), name='mrcnn_class_bn1')

    tdistr_conv2 = tfl.TimeDistributed(tfl.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")

    tdistr_bn2 = tfl.TimeDistributed(tfl.BatchNormalization(trainable=train_bn), name='mrcnn_class_bn2')

    # Classifier head
    mrcnn_class_logits_layer = tfl.TimeDistributed(tfl.Dense(num_classes), name='fpnclf_mrcnn_class_logits')
    mrcnn_probs_layer = tfl.TimeDistributed(tfl.Activation("softmax"), name="fpnclf_mrcnn_class")

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    mrcnn_bbox_fc = tfl.TimeDistributed(tfl.Dense(num_classes * 4, activation='linear'), name='fpnclf_mrcnn_bbox_fc')

    rois, image_meta, feature_maps = inputs
    x = roi_align([rois, image_meta] + feature_maps)
    x = tdistr_bn1(tdistr_conv1(x))
    x = tfl.Activation('relu', name='fpnclf_relu_act1')(x)
    x = tdistr_bn2(tdistr_conv2(x))
    x = tfl.Activation('relu', name='fpnclf_relu_act2')(x)  # -> [None, None, 1, 1, 1024]

    # Fix for several backbones
    if training:
        shared = tfl.Lambda(lambda x: tf.squeeze(tf.squeeze(x, 3), 2),
                            name="fpnclf_pool_squeeze")(x)  # -> [None, None, 1024]
        # shared = tfl.Lambda(lambda x: tf.reshape(x, (batch_size, post_nms_rois_inference, 1024)),
        #                     name="fpnclf_pool_squeeze")(x)
    else:
        shared = tfl.Reshape(target_shape=(post_nms_rois_inference, 1024),
                             name="fpnclf_pool_squeeze")(x)

    mrcnn_class_logits = mrcnn_class_logits_layer(shared)
    mrcnn_probs = mrcnn_probs_layer(mrcnn_class_logits)

    x = mrcnn_bbox_fc(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    if training:
        mrcnn_bbox = tfl.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], num_classes, 4)),
                                name="fpnclf_mrcnn_bbox_reshape")(x)
    else:
        mrcnn_bbox = tfl.Reshape(target_shape=(post_nms_rois_inference, num_classes, 4),
                                 name='fpnclf_mrcnn_bbox_reshape')(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def fpn_mask_graph(inputs, pool_size, num_classes, train_bn):
    roi_align = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")

    fpnmask_conv1 = tfl.Conv2D(256, (3, 3), padding="same", name='fpnmask_conv1')
    tdistr_conv1 = tfl.TimeDistributed(fpnmask_conv1, name="mrcnn_mask_conv1")
    fpnmask_bn1 = tfl.BatchNormalization(trainable=train_bn, name='fpnmask_bn1')
    tdistr_bn1 = tfl.TimeDistributed(fpnmask_bn1, name='mrcnn_mask_bn1')

    fpnmask_conv2 = tfl.Conv2D(256, (3, 3), padding="same", name='fpnmask_conv2')
    tdistr_conv2 = tfl.TimeDistributed(fpnmask_conv2, name="mrcnn_mask_conv2")
    fpnmask_bn2 = tfl.BatchNormalization(trainable=train_bn, name='fpnmask_bn2')
    tdistr_bn2 = tfl.TimeDistributed(fpnmask_bn2, name='mrcnn_mask_bn2')

    fpnmask_conv3 = tfl.Conv2D(256, (3, 3), padding="same", name='fpnmask_conv3')
    tdistr_conv3 = tfl.TimeDistributed(fpnmask_conv3, name="mrcnn_mask_conv3")
    fpnmask_bn3 = tfl.BatchNormalization(trainable=train_bn, name='fpnmask_bn3')
    tdistr_bn3 = tfl.TimeDistributed(fpnmask_bn3, name='mrcnn_mask_bn3')

    fpnmask_conv4 = tfl.Conv2D(256, (3, 3), padding="same", name='fpnmask_conv4')
    tdistr_conv4 = tfl.TimeDistributed(fpnmask_conv4, name="mrcnn_mask_conv4")
    fpnmask_bn4 = tfl.BatchNormalization(trainable=train_bn, name='fpnmask_bn4')
    tdistr_bn4 = tfl.TimeDistributed(fpnmask_bn4, name='mrcnn_mask_bn4')

    fpnmask_deconv = tfl.Conv2DTranspose(256, (2, 2), strides=2, activation="relu", name='fpnmask_convt')
    tdistr_deconv = tfl.TimeDistributed(fpnmask_deconv, name="mrcnn_mask_deconv")

    fpn_conv_ = tfl.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid", name='fpnmask_conv_')
    tdistr_mrcnn_mask = tfl.TimeDistributed(fpn_conv_, name="mrcnn_mask")

    rois, image_meta, feature_maps = inputs

    x = roi_align([rois, image_meta] + feature_maps)
    x = tdistr_bn1(tdistr_conv1(x))
    x = tfl.Activation('relu', name='fpnmask_relu_act1')(x)

    x = tdistr_bn2(tdistr_conv2(x))
    x = tfl.Activation('relu', name='fpnmask_relu_act2')(x)

    x = tdistr_bn3(tdistr_conv3(x))
    x = tfl.Activation('relu', name='fpnmask_relu_act3')(x)

    x = tdistr_bn4(tdistr_conv4(x))
    x = tfl.Activation('relu', name='fpnmask_relu_act4')(x)

    x = tdistr_deconv(x)
    x = tdistr_mrcnn_mask(x)

    return x


class MaskRCNNBackbone:
    def __init__(self, backbone_name, weights, input_shape):
        """
        Mask-RCNN backbones manager
        Args:
            backbone_name: Mask-RCNN backbone name, str
            weights:       Pretrained weights name, str
            input_shape:   Input image shape, list of ints
        """
        super(MaskRCNNBackbone, self).__init__()
        self.backbone_name = backbone_name
        self.weights = weights
        self.input_shape = input_shape
        self.preprocess_input = None
        self.backbone = None
        self._backbone_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
                               'mobilenet', 'mobilenetv2',
                               'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
                               ]

        self.backbone_outputs = {
            'resnet18': ['pooling0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1'],
            'resnet34': ['pooling0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1'],
            'resnet50': ['pooling0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1'],
            'resnet101': ['pooling0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1'],

            'mobilenet': ['conv_pw_1_relu', 'conv_pw_3_relu', 'conv_pw_5_relu',
                          'conv_pw_10_relu', 'conv_pw_13_relu'],
            'mobilenetv2': ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu',
                            'block_13_expand_relu', 'out_relu'],

        }

        self.backbone_outputs.update(
            {f'efficientnetb{i}': ['block2a_activation', 'block3a_expand_activation',
                                   'block4a_expand_activation', 'block6a_expand_activation',
                                   'top_activation'] for i in [0, 1, 2, 3]}
        )

        if self.backbone_name not in self._backbone_list:
            raise NotImplementedError(f'Only {self._backbone_list} backbones. The chosen: {self.backbone_name}')

    def _get_notop_model(self):

        _effnet_mapping = {'efficientnetb0': efn.EfficientNetB0,
                           'efficientnetb1': efn.EfficientNetB1,
                           'efficientnetb2': efn.EfficientNetB2,
                           'efficientnetb3': efn.EfficientNetB3,
                           }
        if 'efficientnet' in self.backbone_name:
            # EfficientNets
            model = _effnet_mapping[self.backbone_name](input_shape=self.input_shape,
                                                        weights=self.weights,
                                                        include_top=False)
        else:
            # ResNets, MobileNets
            model_class, preprocess_input = Classifiers.get(self.backbone_name)
            model = model_class(input_shape=self.input_shape, weights=self.weights, include_top=False)
            self.preprocess_input = preprocess_input

        return model

    def _choose_backbone(self):

        model = self._get_notop_model()
        # Set trainable=False to all batchnorm layers
        bn_layers = [x.name for x in model.layers if 'bn' in x.name]
        for layer_name in bn_layers:
            model.get_layer(layer_name).trainable = False

        self.backbone = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(x).output for x in self.backbone_outputs[self.backbone_name]],
            name=f'backbone_{self.backbone_name}'
        )

        return self.backbone

    def build(self):
        return self._choose_backbone()
