import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import os

output_dir = "data/tensorflow_datasets/cifar10/3.0.2"
# Function to convert numpy images and labels into TFRecord format
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def _int64_feature(value):
    """Returns an int64_list from a int / bool."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_tfrecord(images, labels, filename):
    """Converts a dataset to TFRecord format and writes it to a file."""
    with tf.io.TFRecordWriter(os.path.join(output_dir, filename)) as writer:
        for image, label in zip(images, labels):
            feature = {
                'image': _bytes_feature(image),  # Image as byte string
                'label': _int64_feature(label)  # Label as integer
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create TFRecord files for training, validation, and testing
create_tfrecord(train_images, train_labels, 'cifar10-train.tfrecord')
create_tfrecord(val_images, val_labels, 'cifar10-validation.tfrecord')
create_tfrecord(test_images, test_labels, 'cifar10-test.tfrecord')