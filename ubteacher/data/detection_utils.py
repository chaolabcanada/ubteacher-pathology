# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging

import torchvision.transforms as transforms
from ubteacher.data.transforms.augmentation_impl import GaussianBlur


def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.4)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        #augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2))

        # randomcrop
        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # convert float tensor to int tensor
                transforms.Lambda(lambda x: x.mul(255).byte()),
<<<<<<< HEAD
                transforms.RandomEqualize(p=0.2
                ),
                transforms.RandomAdjustSharpness(0.1, p=0.2
                ),
                transforms.RandomAutocontrast(p=0.2
                ),
=======
                transforms.RandomSolarize(128, p=0.2
                ),
                transforms.RandomEqualize(p=0.2
                ),
                transforms.RandomAdjustSharpness(0.1, p=0.2
                ),
                transforms.RandomAutocontrast(p=0.2
                ),
>>>>>>> e672db9 (split augmenting logic)
                #transforms.RandomApply(transforms.RandomChannelPermutation(), p=0.2
                #),
                transforms.ToPILImage(),    
            ]
        )
        augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)
