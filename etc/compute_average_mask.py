#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2017 Eduardo Valle. All rights reserved.
# eduardovalle.com/ github.com/learningtitans
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import chain
import glob
import os.path
import shutil
import sys

from PIL import Image, ImageMath

version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)


lesions_dir = sys.argv[1]
source_masks_dir = sys.argv[2]
target_masks_dir = sys.argv[3]

def get_id_from_path(path):
  return os.path.splitext(os.path.basename(path))[0]

lesions_glob = glob.iglob(os.path.join(lesions_dir, '*.jpg'))
for lesion_file in lesions_glob:
  lesion = Image.open(lesion_file)
  width, height = lesion.size
  lesion_id = get_id_from_path(lesion_file)
  suffixes = [ 'mask*', '.*', '_*' ]
  masks_glob = chain(*( glob.iglob(os.path.join(source_masks_dir, lesion_id + s + '.png'))
                        for s in suffixes))
  masks = [ (get_id_from_path(mask_file), Image.open(mask_file)) for mask_file in masks_glob ]
  if not masks:
    print('WARNING: No mask found for lesion %s' % lesion_id, file=sys.stderr)
    continue
  ok = True
  for mask_id,mask in masks:
    mask_width, mask_height = mask.size
    if mask_width!=width or mask_height!=height:
      ok = False
      message = ('WARNING: lesion %s and its mask %s have incompatible sizes (expected %dWx%dH found %dWx%dH).' %
            (lesion_id, mask_id, width, height, mask_width, mask_height, ))
      print(message, file=sys.stderr)
  if not ok:
    continue
  n_masks = len(masks)
  _POSSIBLE_VARS = 'abcdefghijklmnopqrstuvwxyz'
  _MAX_MASKS = len(_POSSIBLE_VARS)
  if n_masks>_MAX_MASKS:
    print('WARNING: lesion %s has more than %d masks.' % (lesion_file, _MAX_MASKS), file=sys.stderr)
    n_masks=_MAX_MASKS
    masks=masks[:n_masks]
  average_mask_file = os.path.join(target_masks_dir, '%s_%d.png' % (lesion_id, n_masks))
  if n_masks==1:
    print('copy')
    shutil.copy(os.path.join(source_masks_dir, masks[0][0] + '.png'), average_mask_file)
  else:
    images = [ m[1] for m in masks ]
    image_vars = _POSSIBLE_VARS[:n_masks]
    image_assigns = { v : img for v,img in zip(image_vars,images) }
    image_terms = [ 'float(%s)' % v for v in image_vars ]
    image_formula = 'convert((' + '+'.join(image_terms) + ')/' + str(float(n_masks))+', "L")'
    print(image_formula)
    average_mask = ImageMath.eval(image_formula, **image_assigns)
    average_mask.save(average_mask_file)
