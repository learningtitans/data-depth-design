# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Contributed 2017 Eduardo Valle. eduardovalle.com/ github.com/learningtitans
# flowers.py => skin_lesions.py
"""Provides data for the skin_lesion dataset.

The dataset scripts used to create the dataset can be found at:
prepare_skin_lesions_train.py
prepare_skin_lesions_test.py
convert_skin_lesions_dataset.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import tensorflow as tf

slim = tf.contrib.slim

_FILE_PATTERN = 'skin_lesions_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'image'                : 'A color image of varying size.',
    'label'                : 'General lesion label (0 - nevus, 1 - melanoma, 2 - seborrheic keratosis).',
    'melanoma'             : 'Label for melanoma task (1 if melanoma).',
    'keratosis'            : 'Label for seborrheic keratosis task (1 if keratosis).',
    'id'                   : 'id of the image.',
    'image_type'           : 'Type of image (1 if dermoscopic, 0 if clinical).',
    'diagnosis'            : 'Controlled term for diagnosis from diagnoses-thesaurus.txt.',
    'diagnosis_method'     : 'Method for diagnosis (0.0 - clinic[al examination of image], 1.0 - [clinical] follow-up, 2.0 - histopathology, nan - unknown)',
    'diagnosis_difficulty' : 'Diagnosis difficulty (0.0 - low, 1.0 - medium, 2.0 - high, nan - unknown).',
    'diagnosis_confidence' : 'Diagnosis confidence (0.0 - low, 1.0 - medium, 2.0 - high, nan - unknown).',
    'lesion_thickness'     : 'Lesion thickness (descriptive).',
    'lesion_diameter'      : 'Lesion diameter in mm (or nan if unknown).',
    'lesion_location'      : 'Controlled term for location from locations-thesaurus.txt.',
    'age'                  : 'Age in years (or nan if unknwon).',
    'sex'                  : 'Sex (0.0 - female, 1.0 - male, nan - unknown).',
    'case'                 : 'Case number found in metadata.',
    'alias'                : 'Alias found in metadata (id of aliased image).',
    'semiduplicate'        : 'Semiduplicate group (found by image comparison).',
}


def _keys_to_features():
  return {
      'image/encoded'             : tf.FixedLenFeature((), tf.string,  default_value=''),
      'image/format'              : tf.FixedLenFeature((), tf.string,  default_value='jpg'),
      'class/label'               : tf.FixedLenFeature([], tf.int64,   default_value=tf.zeros([], dtype=tf.int64)),
      'class/melanoma'            : tf.FixedLenFeature([], tf.int64,   default_value=tf.zeros([], dtype=tf.int64)),
      'class/keratosis'           : tf.FixedLenFeature([], tf.int64,   default_value=tf.zeros([], dtype=tf.int64)),
      'meta/id'                   : tf.FixedLenFeature([], tf.string,  default_value=''),
      'meta/image_type'           : tf.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'meta/diagnosis'            : tf.FixedLenFeature((), tf.string,  default_value=''),
      'meta/diagnosis_method'     : tf.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'meta/diagnosis_difficulty' : tf.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'meta/diagnosis_confidence' : tf.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'meta/lesion_thickness'     : tf.FixedLenFeature((), tf.string,  default_value=''),
      'meta/lesion_diameter'      : tf.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'meta/lesion_location'      : tf.FixedLenFeature((), tf.string,  default_value=''),
      'meta/age'                  : tf.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'meta/sex'                  : tf.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'meta/case'                 : tf.FixedLenFeature((), tf.string,  default_value=''),
      'meta/alias'                : tf.FixedLenFeature((), tf.string,  default_value=''),
      'meta/semiduplicate'        : tf.FixedLenFeature((), tf.string,  default_value=''),
  }


def _items_to_handlers():
  return {
      'image'                : slim.tfexample_decoder.Image(),
      'label'                : slim.tfexample_decoder.Tensor('class/label'),
      'melanoma'             : slim.tfexample_decoder.Tensor('class/melanoma'),
      'keratosis'            : slim.tfexample_decoder.Tensor('class/keratosis'),
      'id'                   : slim.tfexample_decoder.Tensor('meta/id'),
      'image_type'           : slim.tfexample_decoder.Tensor('meta/image_type'),
      'diagnosis'            : slim.tfexample_decoder.Tensor('meta/diagnosis'),
      'diagnosis_method'     : slim.tfexample_decoder.Tensor('meta/diagnosis_method'),
      'diagnosis_difficulty' : slim.tfexample_decoder.Tensor('meta/diagnosis_difficulty'),
      'diagnosis_confidence' : slim.tfexample_decoder.Tensor('meta/diagnosis_confidence'),
      'lesion_thickness'     : slim.tfexample_decoder.Tensor('meta/lesion_thickness'),
      'lesion_diameter'      : slim.tfexample_decoder.Tensor('meta/lesion_diameter'),
      'lesion_location'      : slim.tfexample_decoder.Tensor('meta/lesion_location'),
      'age'                  : slim.tfexample_decoder.Tensor('meta/age'),
      'sex'                  : slim.tfexample_decoder.Tensor('meta/sex'),
      'case'                 : slim.tfexample_decoder.Tensor('meta/case'),
      'alias'                : slim.tfexample_decoder.Tensor('meta/alias'),
      'semiduplicate'        : slim.tfexample_decoder.Tensor('meta/semiduplicate'),
  }

def _get_split(split_name, dataset_dir, file_pattern=None, reader=None,
    _items_to_descriptions=None, _keys_to_features=None, _items_to_handlers=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  tf.logging.info('==================================='
                  ' SPLIT %s '
                  '===================================', split_name)
  SPLITS_TO_SIZES  = pickle.load(open(os.path.join(dataset_dir, 'splits_to_sizes.pkl'),  'rb'))
  # CLASSES_TO_SIZES = pickle.load(open(os.path.join(dataset_dir, 'classes_to_sizes.pkl'), 'rb'))

  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = _keys_to_features()

  items_to_handlers = _items_to_handlers()

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = { 0 : 'nevus', 1 : 'melanoma', 2 : 'keratosis' }
  num_classes = 3

  dataset = slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=num_classes,
      labels_to_names=labels_to_names)

  dataset.__dict__['input_field'] = 'image'

  return dataset


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  return  _get_split(split_name, dataset_dir, file_pattern=file_pattern, reader=reader,
      _items_to_descriptions=_ITEMS_TO_DESCRIPTIONS, _keys_to_features=_keys_to_features,
      _items_to_handlers=_items_to_handlers)
