import tensorflow as tf
import os
filenames = []
dataset_dir = "/home/david/datasets/font_data"
def generate_filename_queue(filenames, data_dir, num_epochs=None):
    print("filenames in queue:", filenames)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(data_dir, filenames[i])
    return tf.train.string_input_producer(filenames, num_epochs=num_epochs)

def read_and_decode_single_example(filename_queue):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = tf.TFRecordReader().read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([4096], tf.int64)
        })
    # now return the converted data
    return features['label'], features['image']

def inputs(batch_size=32, reshape_images=True, one_hot=True, num_epochs=None, batch_ids=range(10)):
    filenames = []
    for i in batch_ids:
        filenames.append("font-batch-" + str(i) + ".tfrecords")

    filename_queue = generate_filename_queue(filenames, dataset_dir, num_epochs=num_epochs)
    label, image = read_and_decode_single_example(filename_queue)
    # groups examples into batches randomly
    images_batch, labels_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        capacity=10000*3*batch_size,
        min_after_dequeue=10000)
    if reshape_images:
        images_batch = tf.reshape(images_batch, [batch_size, 64, 64, 1])
    images_batch = (tf.cast(images_batch, tf.float32) / 128.) - 1.
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, 62)
    return images_batch, labels_batch