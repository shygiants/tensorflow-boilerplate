"""
    Dataset encoder for MNIST
"""

import os
from six.moves import urllib
import tempfile
import gzip
import shutil

import tensorflow as tf
import numpy as np
from tflibs.runner import Runner

from datasets.dataset.mnist import MNIST


def download(save_dir, filename):
    save_path = os.path.join(save_dir, filename)
    if tf.gfile.Exists(save_path):
        return save_path
    if not tf.gfile.Exists(save_dir):
        tf.gfile.MakeDirs(save_dir)

    base_url = 'http://yann.lecun.com/exdb/mnist/'
    url = base_url + filename + '.gz'

    _, filepath = tempfile.mkstemp(suffix='.gz')
    tf.logging.info('Downloading {} to {}...'.format(url, filepath))
    urllib.request.urlretrieve(url, filepath)

    with gzip.open(filepath, 'rb') as f_in, \
            tf.gfile.Open(save_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    return save_path


def read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (f.name, rows, cols))


def check_labels_file_header(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))


def fixed_record_reader(filename, record_bytes, header_bytes=0):
    with tf.gfile.Open(filename, 'rb') as f:
        if header_bytes > 0:
            # Abandon header
            f.read(header_bytes)

        while True:
            bs = f.read(record_bytes)
            if bs == '':
                break
            yield np.frombuffer(bs, dtype=np.uint8)


def get_img_lbl(image_fname, label_fname, save_dir):
    img_path = download(save_dir, image_fname)
    lbl_path = download(save_dir, label_fname)

    check_image_file_header(img_path)
    check_labels_file_header(lbl_path)

    images = map(lambda flat_img: np.reshape(flat_img, (28, 28, 1)),
                 fixed_record_reader(img_path, 28 * 28, header_bytes=16))
    labels = list(fixed_record_reader(lbl_path, 1, header_bytes=8))

    return zip(images, labels)


def run(dataset_dir,
        save_dir):
    train_mnist = MNIST(dataset_dir, split='train')
    test_mnist = MNIST(dataset_dir, split='test')

    # Prepare MNIST dataset
    train = get_img_lbl('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', save_dir)
    test = get_img_lbl('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', save_dir)

    def process_fn((i, (img, lbl)), feature_specs):
        def build_example(_id, image, label):
            return {
                '_id': _id.create_with_string('{:05d}.jpg'.format(i)),
                'image': image.create_with_tensor(img),
                'label': label.create_with_index(lbl),
            }

        return build_example(**feature_specs)

    train_mnist.write(list(enumerate(train)), process_fn, num_parallel_calls=16)
    test_mnist.write(list(enumerate(test)), process_fn)


if __name__ == '__main__':
    runner = Runner(no_job_dir=True)
    parser = runner.argparser

    parser.add_argument('--dataset-dir',
                        required=True,
                        type=str,
                        help='The directory where datasets are stored.')
    parser.add_argument('--save-dir',
                        default='/tmp',
                        type=str,
                        help='The directory where MNIST files are stored.')

    runner.run(run)
