# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Contributed 2017 Eduardo Valle. eduardovalle.com/ github.com/learningtitans
# download_and_convert_flowers.py => convert_skin_lesions.py
r"""Converts Melanoma data to TFRecords of TF-Example protos.

This reads the files that make up the Melanoma data and creates three
TFRecord datasets: train, validation, and test. Each TFRecord dataset
is comprised of a set of TF-Example protocol buffers, each of which contain
a single image and label.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)

# Seed for repeatability
_RANDOM_SEED = 0

_DELIMITER = ';'

# Command-line parsing
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train', None,
    'File with the metadata for the train split. '
    'At least one of --train or --test must be present.')

tf.app.flags.DEFINE_string(
    'test', None,
    'File with the metadata for the test split. '
    'At least one of --train or --test must be present.')

tf.app.flags.DEFINE_string(
    'images_dir', None,
    'Directory with all images.')

tf.app.flags.DEFINE_string(
    'masks_dir', None,
    'Directory with all masks. If inform, implies that a skin_lesions_seg dataset '
    'should be created.')

tf.app.flags.DEFINE_string(
    'output_dir', None,
    'Directory to receive the newly created dataset.')

tf.app.flags.DEFINE_integer(
    'samples_per_shard', 1024,
    'The number of elements to store in each dataset shard. '
    'This does not affect the models, just the file storage.')

tf.app.flags.DEFINE_bool(
    'allow_old_python', False, 'The script was not tested on Python 2 and will normally require Python 3.4+, '
    'but this flag allows using older versions (use it at your own risk).')

FLAGS = tf.app.flags.FLAGS


# Copied from deleted dataset_utils.py ===>
def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def float_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def bytes_feature(values):
  if isinstance(values, str):
    values = bytes(values, 'utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
# <===

def to_category(values, *categories, **kwargs):
  default = kwargs.get('default', np.nan)
  if isinstance(values, (tuple, list)):
    return [ categories.index(v) if v in categories else default for v in values ]
  else:
    return categories.index(values) if values in categories else default


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
      feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


class MaskReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes grayscale PNG data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=1)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
      feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards, masks=False):
  output_filename = 'skin_lesions_%s%s_%05d-of-%05d.tfrecord' % (
    'seg_' if masks else '', split_name, shard_id+1, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _diagnosis_to_idx(diagnosis):
  return to_category(diagnosis[:4], '1.1.', '3.2.', '1.5.', default='error')

def _image_to_tfexample(image_data, image_format, height, width, metadata,
      mask_data=None, mask_format=None):
  feature = {
    'image/encoded'             : bytes_feature(image_data),
    'image/format'              : bytes_feature(image_format),
    'class/label'               : int64_feature(_diagnosis_to_idx(metadata[4])),
    'class/melanoma'            : int64_feature(int(metadata[4][:4]=='3.2.')),
    'class/keratosis'           : int64_feature(int(metadata[4][:4]=='1.5.')),
    'height'                    : int64_feature(height),
    'width'                     : int64_feature(width),
    'meta/dataset'              : bytes_feature(metadata[0]),
    'meta/split'                : bytes_feature(metadata[1]),
    'meta/id'                   : bytes_feature(metadata[2]),
    'meta/image_type'           : float_feature(to_category(metadata[3], 'clinical', 'dermoscopic')),
    'meta/diagnosis'            : bytes_feature(metadata[4]),
    'meta/diagnosis_method'     : float_feature(to_category(metadata[5], 'clinic', 'follow-up', 'histopathology')),
    'meta/diagnosis_difficulty' : float_feature(to_category(metadata[6], 'low', 'medium', 'high')),
    'meta/diagnosis_confidence' : float_feature(to_category(metadata[7], 'low', 'medium', 'high')),
    'meta/lesion_thickness'     : bytes_feature(metadata[8]),
    'meta/lesion_diameter'      : float_feature(float(metadata[9]) if metadata[9] else np.nan),
    'meta/lesion_location'      : bytes_feature(metadata[10]),
    'meta/age'                  : float_feature(float(metadata[11]) if metadata[11] else np.nan),
    'meta/sex'                  : float_feature(to_category(metadata[12], 'female', 'male')),
    'meta/case'                 : bytes_feature(metadata[13]),
    'meta/alias'                : bytes_feature(metadata[14]),
    'meta/semiduplicate'        : bytes_feature(metadata[15]),
  }
  if mask_data:
    feature['mask/encoded'] = bytes_feature(mask_data)
    feature['mask/format']  = bytes_feature(mask_format)
  return tf.train.Example(features=tf.train.Features(feature=feature))


def _convert_dataset(split_name, metadata_file, images_dir, masks_dir, dataset_dir):
  """Converts the given images and metadata to a TFRecord dataset.

  Args:
    split_name: The name of the dataset: 'train', or 'test'
    metadata: A list with the dataset metadata
    images_dir: The directory with the input .jpg images
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'test']

  # Get metadata
  #   0  => dataset
  #   1  => split
  #   2  => image (id)
  #   3  => image_type
  #   4  => diagnosis
  #   5  => diagnosis_method
  #   6  => diagnosis_difficulty
  #   7  => diagnosis_confidence
  #   8  => lesion_thickness
  #   9  => lesion_diameter
  #  10  => lesion_location
  #  11  => age
  #  12  => sex
  #  13  => case
  #  14  => alias
  #  15  => semiduplicate

  # Checks and skips header
  metadata = [ m.strip().split(_DELIMITER) for m in open(metadata_file) ]
  metadata = [ [ f.strip() for f in m] for m in metadata ]
  metadata = metadata[1:]

  dataset_size = len(metadata)
  metadata = iter(metadata)
  _NUM_PER_SHARD = FLAGS.samples_per_shard
  num_shards = int(math.ceil(dataset_size / _NUM_PER_SHARD))
  if dataset_size % _NUM_PER_SHARD < int(_NUM_PER_SHARD/3.0):
    num_shards = max(num_shards-1, 1)

  lesion_sizes = [ 0, 0, 0 ]

  with tf.Graph().as_default(), tf.Session('') as session:
    image_reader = ImageReader()
    mask_reader = MaskReader()

    for shard_id in range(num_shards):
      output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards)
      tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)

      start_ndx = shard_id*_NUM_PER_SHARD
      end_ndx   = (shard_id+1)*_NUM_PER_SHARD if shard_id<num_shards-1 else dataset_size
      for i in range(start_ndx, end_ndx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d in %s split' %
          (i+1, dataset_size, shard_id, split_name))
        sys.stdout.flush()

        # Read the image file:
        meta = next(metadata)

        lesion_sizes[_diagnosis_to_idx(meta[4])] += 1

        image_file = os.path.join(images_dir, meta[2]) + '.jpg'
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        image_height, image_width = image_reader.read_image_dims(session, image_data)

        if masks_dir:
          # Assigns the first mask available to the image
          masks_glob = glob.iglob(os.path.join(masks_dir, meta[2]) + '*.png')
          try:
            mask_file = next(masks_glob)
          except StopIteration:
            message = 'no mask found for image %s.' % image_file
            tf.logging.error(message)
            raise RuntimeError(message)
          mask_data = tf.gfile.FastGFile(mask_file, 'rb').read()
          height, width = mask_reader.read_image_dims(session, mask_data)
          if height!=image_height or width!=image_width:
            message = ('image %s and its mask %s have incompatible sizes (expected %dx%d found %dx%d).' %
                  (image_file, mask_file, image_height, image_width, height, width,))
            tf.logging.error(message)
            raise RuntimeError(message)
          example = _image_to_tfexample(image_data, b'jpg', image_height, image_width, meta, mask_data, b'png')
        else:
          # Creates a dataset without masks
          example = _image_to_tfexample(image_data, b'jpg', image_height, image_width, meta)

        tfrecord_writer.write(example.SerializeToString())

      tfrecord_writer.close()
  sys.stdout.write('\n')
  sys.stdout.flush()

  return lesion_sizes


def run(train_file, test_file, images_dir, masks_dir, dataset_dir):
  """Runs the download and conversion operation.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  _EXPECTED_HEADER=('dataset;split;image;image_type;diagnosis;diagnosis_method;'
                    'diagnosis_difficulty;diagnosis_confidence;'
                    'lesion_thickness;lesion_diameter;lesion_location;age;sex;'
                    'case;alias;semiduplicate')
  _EXPECTED_FIELDS=_EXPECTED_HEADER.count(_DELIMITER)+1

  def check_header(m, file):
    if _DELIMITER.join(m[:_EXPECTED_FIELDS])!=_EXPECTED_HEADER:
      tf.logging.fatal('invalid header on metadata file %s' % file)
      sys.exit(1)

  # Sanity check
  if train_file and test_file:
    metadata = [ m.strip().split(_DELIMITER) for m in open(train_file) ]
    check_header(metadata[0], train_file)
    train_ids = [ m[2].strip() for m in metadata[1:] ]
    metadata = [ m.strip().split(_DELIMITER) for m in open(test_file) ]
    check_header(metadata[0], test_file)
    test_ids = [ m[2].strip() for m in metadata[1:] ]
    common = set(train_ids).intersection(set(test_ids))
    if common:
      tf.logging.fatal('train and test files have common ids %s' % ' '.join(common))
      sys.exit(1)

  # Convert the training and validation sets.
  if train_file:
    train_sizes = _convert_dataset('train', train_file, images_dir, masks_dir, dataset_dir)
  else:
    train_sizes = [ 0, 0, 0 ]

  if test_file:
    test_sizes = _convert_dataset('test', test_file, images_dir, masks_dir, dataset_dir)
  else:
    test_sizes = [ 0, 0, 0 ]

  # Saves classes and split sizes
  CLASSES_TO_SIZES = { 'nevus'     : train_sizes[0]+test_sizes[0],
                       'melanoma'  : train_sizes[1]+test_sizes[1],
                       'keratosis' : train_sizes[2]+test_sizes[2] }

  tf.logging.info(str(CLASSES_TO_SIZES))
  pickle.dump(CLASSES_TO_SIZES, open(os.path.join(dataset_dir, 'classes_to_sizes.pkl'), 'wb'))

  SPLITS_TO_SIZES = { 'train' : sum(train_sizes),
                      'test'  : sum(test_sizes) }
  tf.logging.info(str(SPLITS_TO_SIZES))
  pickle.dump(SPLITS_TO_SIZES, open(os.path.join(dataset_dir, 'splits_to_sizes.pkl'),  'wb'))


def main(unparsed):

  this_app_path = unparsed[0]
  unparsed = unparsed[1:]
  if unparsed:
    raise ValueError('Unrecognized arguments: %s' % ' '.join(unparsed))

  if not (FLAGS.test or FLAGS.train):
    raise ValueError('You must supply one of --train or --test')

  if not FLAGS.images_dir:
    raise ValueError('You must specify the images directory in --images_dir')

  if not FLAGS.output_dir:
    raise ValueError('You must specify the dataset directory in --output_dir')

  _MIN_SAMPLES_PER_SHARD=256
  if FLAGS.samples_per_shard<_MIN_SAMPLES_PER_SHARD:
    raise ValueError('The value of --samples_per_shard must be above %d' % _MIN_SAMPLES_PER_SHARD)

  run(FLAGS.train, FLAGS.test, FLAGS.images_dir, FLAGS.masks_dir, FLAGS.output_dir)

  return 0


if __name__ == '__main__':
  tf.app.run()
