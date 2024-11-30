import tensorflow_datasets as tfds

data_dir = "data/tensorflow_datasets/"
cifar10_builder = tfds.builder(name="cifar10", data_dir=data_dir)
cifar10_builder.download_and_prepare()