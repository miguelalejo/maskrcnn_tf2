import datetime
import hashlib
import os

import tensorflow as tf

from layers import losses
from preprocess import preprocess


def train_model(model, train_dataset, val_dataset, config, weights_path=None, logdir=None):
    """

    Args:
        model:         MaskRCNN model instance, can be subclassed or functional implementation
        train_dataset: Training dataset class
        val_dataset    Validation dataset class
        config:        General config for MaskRCNN
        weights_path:  MaskRCNN checkpoint weights that will be set for a model.
                       Based on this weights initial_epoch for tf.keras.Model fit method will be corrected.
        logdir:        Logging directory for training restore

    Returns: None

    """

    # Check batch_size and images_per_gpu
    if config['images_per_gpu'] != config['batch_size']:
        im_per_gpu = config['images_per_gpu']
        bs = config['batch_size']
        raise Exception(f'images_per_gpu: {im_per_gpu} is not equal to batch_size: {bs} in config.')

    # Initialize DataLoaders
    train_dataloader = preprocess.DataLoader(train_dataset,
                                             shuffle=True,
                                             name='train',
                                             **config
                                             )
    val_dataloader = preprocess.DataLoader(val_dataset,
                                           shuffle=True,
                                           name='val',
                                           **config
                                           )

    # Workaround for multiprocessing with tf.keras.Sequence dataloader wrapped in tf.data.Dataset
    def make_gen_callable(_gen):
        def gen():
            for inputs, outputs in _gen:
                if config['use_rpn_rois']:
                    batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
                    batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = inputs

                    yield batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks
                else:
                    batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
                    batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, random_rois = inputs

                    yield batch_images, batch_images_meta, batch_rpn_match, batch_rpn_bbox, \
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, random_rois

        return gen

    out_len = 7
    if not config['use_rpn_rois'] and not config['random_rois']:
        raise ValueError(f"""Set random_rois>0 for use_rpn_rois={config['use_rpn_rois']} option""")

    if not config['use_rpn_rois'] and config['random_rois']:
        out_len = 8

    train_datagen = tf.data.Dataset.from_generator(generator=make_gen_callable(train_dataloader),
                                                   output_types=tuple(tf.float32 for _ in range(out_len)))
    val_datagen = tf.data.Dataset.from_generator(generator=make_gen_callable(val_dataloader),
                                                 output_types=tuple(tf.float32 for _ in range(out_len)))

    train_datagen, val_datagen = train_datagen.repeat(), val_datagen.repeat()
    if config['use_prefetch']:
        train_datagen = train_datagen.prefetch(config['prefetch_buff_size'])
        val_datagen = val_datagen.prefetch(config['prefetch_buff_size'])

    """
    Initialize losses:
    rpn_class_loss :   How well the Region Proposal Network separates background with objects
    rpn_bbox_loss :    How well the RPN localize objects
    mrcnn_bbox_loss :  How well the Mask RCNN localize objects
    mrcnn_class_loss : How well the Mask RCNN recognize each class of object
    mrcnn_mask_loss :  How well the Mask RCNN segment objects

    """
    losses_dict = {'rpn_class_loss': losses.RPNClassLoss(),
                   'rpn_bbox_loss': losses.RPNBboxLoss(images_per_gpu=config['images_per_gpu']),
                   'mrcnn_class_loss': losses.MRCNNClassLoss(batch_size=config['batch_size']),
                   'mrcnn_bbox_loss': losses.MRCNNBboxLoss(num_classes=config['num_classes']),
                   'mrcnn_mask_loss': losses.MRCNNMaskLoss(),
                   }

    optimizer = get_optimizer(config['optimizer_kwargs'])
    model.compile(optimizer=optimizer, losses_dict=losses_dict, run_eagerly=True)

    # Load weights for MaskRCNN created previously during training.
    bbone = config['backbone']
    tboard_model_folder = f"maskrcnn_{bbone}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_logdir = os.path.join('..', 'logs', 'scalars', tboard_model_folder)
    initial_epoch = 0
    if weights_path:
        model.load_weights(weights_path)
        initial_epoch = int(weights_path.split('cp-')[-1].split('.')[0])
        print(f'\nLoaded weights: {weights_path}\nInitial epoch: {initial_epoch}')

        if logdir:
            tensorboard_logdir = logdir

    # Prepare training callbacks list
    model_md5_config = hashlib.md5(config.__repr__().encode()).hexdigest()
    checkpoint_path = os.path.join(config['callback']['checkpoints_dir'], tboard_model_folder, 'checkpoints',
                                   'maskrcnn_' + config['backbone'] + f'_{model_md5_config}' + '_cp-{epoch:04d}.ckpt')
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss_sum',
            save_best_only=config['callback']['save_best_only'],
            save_weights_only=config['callback']['save_weights_only'],
            save_freq='epoch',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss_sum',
            factor=config['callback']['reduce_lr_on_plateau'],
            patience=config['callback']['reduce_lr_on_plateau_patience'],
        ),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir,
                                       histogram_freq=config['callback']['histogram_freq'],
                                       profile_batch=config['callback']['profile_batch'],
                                       )
    ]

    model.fit(train_datagen,#train_datagen.repeat(),
              steps_per_epoch=train_dataloader.steps_per_epoch,
              validation_data=val_datagen,#val_datagen.repeat(),
              validation_steps=val_dataloader.steps_per_epoch,
              epochs=config['epochs'],
              initial_epoch=initial_epoch,
              callbacks=callbacks_list,
              verbose=True,
              use_multiprocessing=config['use_multiprocessing'],
              workers=config['workers'],
              max_queue_size=int(config['queue_multiplier'] * config['batch_size']),
              )


def get_optimizer(kwargs):
    """
    Select necessary optimizer.
    Args:
        kwargs: dict for a specific model optimizer

    Returns: Optimizer class from tf.keras.optimizers

    """
    opt_name = kwargs['name'].lower()
    if opt_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(**kwargs)
    elif opt_name == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(**kwargs)
    elif opt_name == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(**kwargs)
    elif opt_name == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(**kwargs)
    elif opt_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(**kwargs)
    elif opt_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(**kwargs)
    elif opt_name == 'ftrl':
        optimizer = tf.keras.optimizers.Ftrl(**kwargs)
    else:
        raise NotImplementedError('Only sgd, adam, adamax, adadelta, adagrad, rmsprop, ftrl optimizers are added.')
    return optimizer
