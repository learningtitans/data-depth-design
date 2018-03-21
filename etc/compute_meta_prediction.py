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

import argparse
import logging
import sys

import numpy as np


version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('compute_meta_prediction')


Titans_Header = [ 'id', 'truth', 'nevus[0]', 'melanoma[1]', 'keratosis[2]', 'prediction' ]

def auto_parse_predictions_file(predictions_file):
  # This is needed to deal with a bug in an early version of
  # predict_image_classifier.py and predict_svm_layer.py
  def _format_id(image_id):
    image_id = image_id.strip()
    if image_id[:2] == "b'" and image_id[-1:] == "'":
      return image_id[2:-1]
    else:
      return image_id
  submission = [ [ f.strip() for f in line.strip().split(',') ] for line in predictions_file ]
  if submission[0] == Titans_Header:
    # Titans format
    submission = [ (_format_id(s[0]), float(s[3]), float(s[4]), float(s[2]),) for s in submission[1:] ]
  else:
    # ISBI format
    submission = [ (_format_id(s[0]), float(s[1]), float(s[2]), np.nan,) for s in submission ]
  return submission # id, prediction melanoma, prediction keratosis, prediction nevus


def merge_predictions(meta_predictions, n_meta, new_predictions, n_new, pool_method):
  if pool_method == 'avg':
    # The average must be weighted by the number of samples represented
    # by each vector
    meta_predictions = (meta_predictions*n_meta) + (new_predictions*n_new)
  elif pool_method == 'max':
    meta_predictions = np.maximum(meta_predictions, new_predictions)
  elif  pool_method == 'xtrm':
    meta_predictions_xtrm = np.abs(meta_predictions - 0.5)
    new_predictions_xtrm  = np.abs(new_predictions - 0.5)
    new_wins = new_predictions_xtrm>meta_predictions_xtrm
    meta_predictions[new_wins] = new_predictions[new_wins]
  else:
    assert False

  # Renormalizes predictions
  n_totals = np.sum(meta_predictions, axis=1)
  n_totals[n_totals == 0.0 ] = 1. # If all predictions are zero, keep them like that
  return (meta_predictions.T/n_totals).T


def run(partial_predictions, pool_method, return_table=True, return_array=False):
  assert return_table or return_array

  first = True
  images = []
  n_images = 0
  meta_predictions = None
  for predictions_filename in partial_predictions:
    with open(predictions_filename, 'rt') as predictions_file:
      try:
        predictions = auto_parse_predictions_file(predictions_file)
      except (IndexError, ValueError) as e:
        log.error("Error on file %s: %s", predictions_filename, str(e))
        raise
    if first:
      images = tuple(( f[0] for f in predictions ))
      n_images = len(images)
      meta_predictions = np.array([ f[1:4] for f in predictions ], dtype=np.float)
      first = False
    else:
      if images != tuple(( f[0] for f in predictions )):
        msg = "file %s' image ids do not match with previous predictions files'" % predictions_filename
        logging.fatal(msg)
        raise ValueError(msg)
      else :
        new_predictions = np.asarray([ f[1:4] for f in predictions ], dtype=np.float)
        if pool_method == 'avg':
          meta_predictions = meta_predictions+new_predictions
        elif pool_method == 'max':
          meta_predictions = np.maximum(meta_predictions, new_predictions)
        elif  pool_method == 'xtrm':
          meta_predictions_xtrm = np.abs(meta_predictions - 0.5)
          new_predictions_xtrm  = np.abs(new_predictions - 0.5)
          new_wins = new_predictions_xtrm>meta_predictions_xtrm
          meta_predictions[new_wins] = new_predictions[new_wins]
        else:
          assert False

  # If any of the predictions is NaN, at least of the files is in isbi format
  # print(meta_predictions)
  if np.any(np.isnan(meta_predictions)):
    # Removes nevus column
    meta_predictions = np.delete(meta_predictions, 2, axis=1)
    isbi = True
  else:
    isbi = False
  # print(meta_predictions)

  # Renormalize predictions
  n_totals = np.sum(meta_predictions, axis=1)
  n_totals[n_totals == 0.0 ] = 1. # If all predictions are zero, keep them like that
  meta_predictions = (meta_predictions.T/n_totals).T

  if return_array and not return_table:
    return meta_predictions
  elif return_table:
    results = [ [images[i]]+list(meta_predictions[i,:]) for i in range(n_images) ]
    if return_array:
      return results, isbi, meta_predictions
    else:
      return results, isbi



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--partial_predictions', required=True,
      help='List of files with the partial predictions to be merged in the meta-prediction.')
  parser.add_argument('--output_file', type=argparse.FileType('w'), required=True,
     help='output file with the meta-predictions, in isbi challenge format (default=stdout).')
  parser.add_argument('--pool_method', type=str, choices=('avg', 'max', 'xtrm',), default='avg',
      help='technique used to pool the predictions: avg (default), max, xtrm')
  parser.add_argument('--verbose',
      help='increase output verbosity', action='store_true')
  parser.add_argument('--allow_old_python',
      help='The script was not tested on Python 2 and will normally require Python 3.4+, '
        'but this flag allows using older versions (use it at your own risk).', action='store_true')
  args = parser.parse_args()

  if args.verbose:
    log.setLevel(logging.DEBUG)

  partial_predictions = [ f.strip() for f in args.partial_predictions.split(',') if f.strip()!='' ]
  meta_predictions, isbi = run(partial_predictions, args.pool_method)

  if isbi:
    for p in meta_predictions:
      print(p[0], p[1], p[2], sep=',', file=args.output_file)
  else:
    print(', '.join(Titans_Header), file=args.output_file)
    for p in meta_predictions:
      print(p[0], p[1], p[2], p[3], sep=', ', file=args.output_file)

if __name__ == '__main__':
  main()
