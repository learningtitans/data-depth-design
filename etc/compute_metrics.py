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
import sklearn.metrics

version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger('compute_metrics')


# This must be coherent with datasets/convert_skin_lesions.py
def _diagnosis_to_idx(diagnosis):
  return ('1.1.', '3.2.', '1.5.',).index(diagnosis[:4])


def parse_metadata_file(metadata_file):
  truth = [ line.strip().split(';') for line in metadata_file ]
  truth_header = 'dataset;split;image;image_type;diagnosis;diagnosis_method;' \
                 'diagnosis_difficulty;diagnosis_confidence;lesion_thickness;' \
                 'lesion_diameter;lesion_location;age;sex;case;alias;semiduplicate'
  if ';'.join(truth[0]) != truth_header:
    log.fatal('Invalid header on --metadata file!')
    sys.exit(1)
  truth = { t[2] : _diagnosis_to_idx(t[4]) for t in truth[1:] }
  return truth


def parse_predictions_file(predictions_file, predictions_format):
  # This is needed to deal with a bug in an early version of
  # predict_image_classifier.py and predict_svm_layer.py
  def _format_id(image_id):
    image_id = image_id.strip()
    if image_id[:2] == "b'" and image_id[-1:] == "'":
      return image_id[2:-1]
    else:
      return image_id
  submission = [ line.strip().split(',') for line in predictions_file ]
  if predictions_format == 'isbi':
    submission = [ (_format_id(s[0]), float(s[1]), float(s[2]), None,) for s in submission ]
  else:
    submission = [ (_format_id(s[0]), float(s[3]), float(s[4]), s[1],) for s in submission[1:] ]
  return submission


def compute_metrics(truth, predictions):
  # Melanoma metrics
  np_scores    = np.asarray([ s[1] for s in predictions ])
  np_decisions = np.asarray([ 1 if s[1]>0.5 else 0 for s in predictions ])
  np_labels    = np.asarray([ 1 if t==1 else 0 for t in truth ])
  m_ap  = sklearn.metrics.average_precision_score(np_labels, np_scores)
  m_auc = sklearn.metrics.roc_auc_score(np_labels, np_scores)
  m_cm  = sklearn.metrics.confusion_matrix(np_labels, np_decisions)
  m_tn, m_fp, m_fn, m_tp = m_cm.ravel()
  m_tpr = m_tp / (m_tp+m_fn)
  m_fpr = m_fp / (m_fp+m_tn)

  # Keratosis AUC
  np_scores    = np.asarray([ s[2] for s in predictions ])
  np_decisions = np.asarray([ 1 if s[2]>0.5 else 0 for s in predictions ])
  np_labels    = np.asarray([ 1 if t==2 else 0 for t in truth ])
  k_ap  = sklearn.metrics.average_precision_score(np_labels, np_scores)
  k_auc = sklearn.metrics.roc_auc_score(np_labels, np_scores)
  k_cm  = sklearn.metrics.confusion_matrix(np_labels, np_decisions)
  k_tn, k_fp, k_fn, k_tp = k_cm.ravel()
  k_tpr = k_tp / (k_tp+k_fn)
  k_fpr = k_fp / (k_fp+k_tn)
  # Average combined metrics
  isbi_auc = (m_auc + k_auc) / 2.

  return (m_ap, m_auc, m_tn, m_fp, m_fn, m_tp, m_tpr, m_fpr,
          k_ap, k_auc, k_tn, k_fp, k_fn, k_tp, k_tpr, k_fpr, isbi_auc,)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--metadata_file', type=argparse.FileType('rt'),
      help='File with the dataset metadata, containing the ground truth, mandatory if --predictions_format=isbi.')
  parser.add_argument('--predictions_file', type=argparse.FileType('rt'), required=True,
      help='File with the predictions.')
  parser.add_argument('--predictions_format', choices=['isbi', 'titans'], default='titans',
      help='Format of the prediction file.')
  parser.add_argument('--metrics_file', type=argparse.FileType('wt'),
      help='File to receive the output metrics.')
  parser.add_argument('--verbose', help='increase output verbosity',
                      action='store_true')
  parser.add_argument('--allow_old_python',
      help='The script was not tested on Python 2 and will normally require Python 3.4+, '
      'but this flag allows using older versions (use it at your own risk).', action='store_true')
  args = parser.parse_args()

  if args.verbose:
    log.setLevel(logging.DEBUG)

  predictions = parse_predictions_file(
    args.predictions_file, args.predictions_format)

  if args.metadata_file is None:
    if args.predictions_format == 'isbi':
      log.fatal('--predictions_format=isbi makes --metadata_file mandatory')
      sys.exit(1)
    truth = [ s[2] for s in predictions ]
  else:
    truth = parse_metadata_file(args.metadata_file)
    truth = [ truth[s[0]] for s in predictions ]

  metrics = compute_metrics(truth, predictions)

  # Print results
  output_file = args.metrics_file or sys.stdout
  print('m_ap;m_auc;m_tn;m_fp;m_fn;m_tp;m_tpr;m_fpr;'
        'k_ap;k_auc;k_tn;k_fp;k_fn;k_tp;k_tpr;k_fpr;isbi_auc',
        file=output_file)
  print(';'.join((str(m) for m in metrics)), file=output_file)


if __name__ == '__main__':
  main()
