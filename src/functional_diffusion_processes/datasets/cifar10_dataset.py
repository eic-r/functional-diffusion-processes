import abc
import logging
from typing import Any

import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig

from .dataset_utils import central_crop, resize_small
from .image_dataset import ImageDataset

pylogger = logging.getLogger(__name__)


class CIFAR10Dataset(ImageDataset, abc.ABC):
    """Cifar-10 dataset class for loading and preprocessing the Cifar-10 dataset.

    Inherits from the ImageDataset class and provides specific implementations for loading
    and preprocessing the Cifar-10 dataset.

    Attributes:
        dataset_builder (tfds.core.DatasetBuilder): Builder for the Cifar-10 dataset.
    """

    def __init__(self, data_config: DictConfig, split: str, evaluation: bool = False) -> None:
        """Initialize the CIFAR10Dataset class.

        Initializes the dataset class by calling the super class constructor and sets up the dataset builder
        for the Cifar-10 dataset.

        Args:
            data_config (DictConfig): Configuration parameters for the dataset.
            split (str): Specifies which split of the dataset to load, e.g., 'train', 'validation', or 'test'.
            evaluation (bool): Indicates whether the dataset is for evaluation purposes.

        """
        super().__init__(data_config, split, evaluation)
        self.dataset_builder = tfds.builder(name="cifar10", data_dir=self.data_config.data_dir)

    def _resize_op(self, image: Any, size: int) -> Any:
        """Resizes the input image to the specified size and normalizes its values to the range [0,1].

        Args:
            image (Any): A tensor representing the input image.
            size (int): The target size for each dimension of the output image.

        Returns:
            Any: A tensor representing the resized and normalized image.
        """
        # convert to range [0,1]
        pylogger.info("Converting image to range [0,1]...")
        image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
        # resize to size
        pylogger.info("Resizing image to size {}...".format(size))

        image = tf.image.resize(images=image, size=[size, size])

        return image
