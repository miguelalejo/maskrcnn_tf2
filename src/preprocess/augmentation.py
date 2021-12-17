import albumentations as img_album


def get_training_augmentation(weather=False, flips=True, extend_aug=False):
    """
    Training augmentation pipeline
    Args:
        weather:     bool, use weather augmentation or not
        flips:       bool, use horizontal and vertical flipping or not
        extend_aug:  bool, use additional augmentation operations

    Returns: albumentations augmentation pipeline

    """
    base_transform_list = [
        img_album.GaussianBlur(p=0.5),
        img_album.Rotate(limit=(10, 270)),
        img_album.MultiplicativeNoise(multiplier=(0.5, 1.2)),
        img_album.ChannelShuffle(p=0.5),
    ]

    if flips:
        flip_transform = [img_album.OneOf([img_album.HorizontalFlip(p=0.5),
                                           img_album.VerticalFlip(p=0.5)]
                                          )
                          ]
        base_transform_list.extend(flip_transform)

    if weather:
        weather_transform = [img_album.RandomSnow(p=0.3),
                             img_album.RandomRain(p=0.3),
                             img_album.RandomFog(p=0.2),
                             img_album.RandomSunFlare(p=0.2)
                             ]
        base_transform_list.extend(weather_transform)

    train_transform = [img_album.OneOf(base_transform_list, p=0.5)]

    if extend_aug:
        aug_extensions = [
            img_album.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            img_album.IAAAdditiveGaussianNoise(p=0.2),
            img_album.IAAPerspective(p=0.5),
            img_album.CLAHE(p=0.5),
            img_album.RandomBrightness(p=0.5),
            img_album.RandomGamma(p=0.5),
            img_album.IAASharpen(p=0.5),
            img_album.Blur(blur_limit=3, p=0.5),
            img_album.MotionBlur(blur_limit=3, p=0.5),
            img_album.RandomContrast(p=0.5),
            img_album.HueSaturationValue(p=0.5),

        ]
        train_transform.append(img_album.OneOf(aug_extensions, p=0.5))

    train_transform.append(img_album.Lambda(mask=round_clip_0_1))

    return img_album.Compose(train_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data preprocessing function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        img_album.Lambda(image=preprocessing_fn),
    ]
    return img_album.Compose(_transform)


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)
