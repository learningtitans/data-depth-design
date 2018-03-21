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
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import image_ops

from . import skin_lesions

slim = tf.contrib.slim


class Composite(slim.tfexample_decoder.ItemHandler):
  """An ItemHandler that decodes a parsed Tensor as a composite of
     RGB jpeg image and grayscale PNG mask."""

  def __init__(self,
               image_key=None,
               mask_key=None,
               dtype=dtypes.uint8):
    """Initializes the image.
    Args:
      image_key: the name of the TF-Example feature in which the encoded image
        is stored.
      dtype: images will be decoded at this bit depth. Different formats
        support different bit depths.
          See tf.image.decode_image,
              tf.decode_raw,
    """
    if not image_key:
      image_key = 'image/encoded'
    if not mask_key:
      mask_key = 'mask/encoded'

    super(Composite, self).__init__([image_key, mask_key])
    self._image_key = image_key
    self._mask_key = mask_key
    self._channels = 4
    self._dtype = dtype

  def tensors_to_item(self, keys_to_tensors):
    """See base class."""
    image_buffer = keys_to_tensors[self._image_key]
    mask_buffer = keys_to_tensors[self._mask_key]
    return self._decode(image_buffer, mask_buffer)

  def _decode(self, image_buffer, mask_buffer):
    """Decodes the image buffer.
    Args:
      image_buffer: The tensor representing the encoded image tensor.
      mask_buffer: The tensor representing the encoded mask tensor.
    Returns:
      A tensor that represents decoded image of self._shape, or
      (?, ?, self._channels) if self._shape is not specified.
    """
    image = image_ops.decode_image(image_buffer, 3) # Expected to be in jpg
    mask = image_ops.decode_image(mask_buffer, 1)   # Expected to be in png
    composite = tf.stack([image, mask], axis = -1)
    assert  tf.shape(composite)[2]==4
    return composite


_get_split = skin_lesions._get_split

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  dataset =  _get_split(split_name, dataset_dir, file_pattern=file_pattern, reader=reader,
      _items_to_descriptions=_ITEMS_TO_DESCRIPTIONS, _keys_to_features=_keys_to_features,
      _items_to_handlers=_items_to_handlers)
  dataset.__dict__['input_field'] = 'composite'
  return dataset


_FILE_PATTERN = 'skin_lesions_seg_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = skin_lesions._ITEMS_TO_DESCRIPTIONS.copy()
_ITEMS_TO_DESCRIPTIONS['mask']      = 'Grayscale segmentation mask'
_ITEMS_TO_DESCRIPTIONS['composite'] = '4-plane composite of RGB image + mask'

def _keys_to_features():
  keys_to_features = skin_lesions._keys_to_features()
  keys_to_features['image/encoded'] = tf.FixedLenFeature((), tf.string, default_value='')
  keys_to_features['image/format']  = tf.FixedLenFeature((), tf.string, default_value='jpg')
  keys_to_features['mask/encoded']  = tf.FixedLenFeature((), tf.string, default_value='')
  keys_to_features['mask/format']   = tf.FixedLenFeature((), tf.string, default_value='png')
  return keys_to_features

def _items_to_handlers():
  items_to_handlers = skin_lesions._items_to_handlers()
  items_to_handlers['image']     = slim.tfexample_decoder.Image()
  items_to_handlers['mask']      = slim.tfexample_decoder.Image(image_key='mask/encoded', format_key='mask/format')
  items_to_handlers['composite'] = Composite()
  return items_to_handlers
