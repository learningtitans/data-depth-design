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
import glob
import itertools
import logging
import os.path
import sys

import numpy as np
import pandas as pd
import sklearn.metrics

version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test_all_meta_predictions')

def run(ground_truth, partial_dir, partial_table, partial_template, output_file, pool_methods=('avg',),
    step=32, max_steps=None, sort_datasets=('random',), check_datasets=('0',), metric='isbi_auc'):

  assert metric=='isbi_auc'

  experiments = itertools.product(pool_methods, sort_datasets, check_datasets)
  print('pool_method', 'sort_dataset', 'check_dataset', 'n_meta', 'm_auc', 'k_auc', 'isbi_auc', sep=',', file=output_file)

  n_randoms = sum([ 1 if d=='random' else 0 for d in sort_datasets])
  seq_randoms = 0

  for pool_method, sort_dataset, check_dataset in experiments:
    if sort_dataset=='random':
      order = partial_table[partial_table["j"]==0]
      order = order.sample(frac=1)
      seq_randoms += 1
    else:
      order = partial_table[partial_table["j"]==int(sort_dataset)]
      order = order.sort_values(by=metric, ascending=False)
    check_dataset_j = int(check_dataset)

    def grouper(iterable, n):
      "Collect data into fixed-length chunks or blocks"
      # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
      args = [iter(iterable)] * n
      return itertools.zip_longest(fillvalue=None, *args)

    n_meta = 0
    first = True
    truth = None
    m_labels = None
    k_labels = None
    meta_predictions = None
    for chunk in grouper(order.iterrows(), step):

      partial_filenames = []
      for index_and_row in chunk:
        if index_and_row is None:
          break
        _, row = index_and_row
        partial_filename = partial_template.replace('[[[-D1_N-]]]', str(int(row['D1_N'])))
        partial_filename = partial_filename.replace('[[[-D3_N-]]]', str(int(row['D3_N'])))
        partial_filename = partial_filename.replace('[[[-D4_N-]]]', str(check_dataset_j))
        candidates = glob.glob(os.path.join(partial_dir, partial_filename))
        if len(candidates)!=1:
          log.error('ERROR: globbing for template %s found %d files: %s; this experiment will be ignored',
              partial_filename, len(candidates), ', '.join(candidates))
          continue
        partial_filenames.append(candidates[0])

      n_new = len(partial_filenames)
      if first:
        new_predictions, _, meta_predictions = compute_meta_prediction.run(partial_filenames, pool_method, return_table=True, return_array=True)
        truth = [ ground_truth[s[0]] for s in new_predictions ]
        m_labels = np.asarray([ 1 if t==1 else 0 for t in truth ])
        k_labels = np.asarray([ 1 if t==2 else 0 for t in truth ])
        first = False
      else:
        new_predictions  = compute_meta_prediction.run(partial_filenames, pool_method, return_table=False, return_array=True)
        meta_predictions = compute_meta_prediction.merge_predictions(meta_predictions, n_meta, new_predictions, n_new, pool_method)
      n_meta += n_new

      # Metrics
      m_scores = meta_predictions[:,0]
      m_auc    = sklearn.metrics.roc_auc_score(m_labels, m_scores)
      k_scores = meta_predictions[:,1]
      k_auc    = sklearn.metrics.roc_auc_score(k_labels, k_scores)

      # Average combined metrics
      isbi_auc = (m_auc + k_auc) / 2.

      if sort_dataset == 'random' and n_randoms>1:
        sort_dataset += str(seq_randoms)

      print(pool_method, sort_dataset, check_dataset, n_meta, m_auc, k_auc, isbi_auc, sep=',', file=output_file)
      print(pool_method, sort_dataset, check_dataset, n_meta, m_auc, k_auc, isbi_auc, sep=',')

      if max_steps is not None and n_meta>=max_steps:
        break



def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--metadata_file', type=argparse.FileType('rt'), required=True,
      help='File with the dataset metadata, containing the ground truth.')
  parser.add_argument('--partial_dir', required=True,
      help='Directory with the files with the partial predictions to be merged in the meta-prediction.')
  parser.add_argument('--partial_table', required=True,
      help='File with the metrics for the partial predictions.')
  parser.add_argument('--partial_template', default="deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].test.[[[-D4_N-]]].index.*.*.results.txt",
      help='File with the metrics for the partial predictions.')
  # parser.add_argument('--select', default="h==-1")
  #     help='Selection clause to apply to the metrics table before running the experiment.')
  parser.add_argument('--output_file', type=argparse.FileType('w'), required=True,
     help='Output file with the performance of the differente meta predictions.')
  parser.add_argument('--pool_methods', type=str, default='avg',
      help='Comma-separated list of the pool methods to evaluate.')
  parser.add_argument('--step', type=int, default=32,
      help='Increase step for the number of partial predictions in the meta-prediction.')
  parser.add_argument('--max', type=int, default=None,
      help='Maximum number of partial predictions in the meta-prediction, omit to use all meta_predictions.')
  parser.add_argument('--sort_datasets', type=str, default='random',
      help='Comma-separated list of the test datasets used to sort the techniques, or random.')
  parser.add_argument('--check_datasets', type=str, default='2',
      help='Comma-separated list of the test datasets used to measure the meta-prediction performance.')
  parser.add_argument('--metric', type=str, default='isbi_auc',
      help='Metric used to sort the techniques and measure the meta-prediction performance.')
  parser.add_argument('--verbose',
      help='increase output verbosity', action='store_true')
  parser.add_argument('--allow_old_python',
      help='The script was not tested on Python 2 and will normally require Python 3.4+, '
        'but this flag allows using older versions (use it at your own risk).', action='store_true')
  args = parser.parse_args()

  if args.verbose:
    log.setLevel(logging.DEBUG)

  ground_truth = compute_metrics.parse_metadata_file(args.metadata_file)

  partial_table = pd.read_csv(args.partial_table, sep=";") # TODO: dtype

  # For the moment the selection clause is fixed and removes all experiments with SVM
  partial_table = partial_table[partial_table['h']==-1]

  pool_methods = [ f.strip() for f in args.pool_methods.split(',') if f.strip()!='' ]

  step = int(args.step)

  if step <= 0:
    log.fatal("--step must be >=0")
    sys.exit(1)

  sort_datasets  = [ f.strip() for f in args.sort_datasets.split(',') if f.strip()!='' ]
  ok = False
  try:
    ok = all([ d=='random' or int(d)>=0 for d in sort_datasets ])
  except ValueError:
    pass
  if not ok:
    log.fatal("--sort_datasets must be a comma-separated list of non-negative integers or 'random'")
    sys.exit(1)


  check_datasets = [ f.strip() for f in args.check_datasets.split(',') if f.strip()!='' ]
  ok = False
  try:
    ok = all([ int(d)>=0 for d in check_datasets ])
  except ValueError:
    pass
  if not ok:
    log.fatal("--check_datasets must be a comma-separated list of non-negative integers")
    sys.exit(1)

  if args.max is not None and args.max <0:
    log.fatal("--max must be a non-negative integer")
    sys.exit(1)

  # For the moment supports only "isbi_auc"
  # if args.metric not in results_table.columns.values:
  #   log.fatal("--metric must be one f the headers of the partial_table" % )
  #   sys.exit(1)
  if args.metric!="isbi_auc":
    log.fatal("--metric %s is currently unsupported", args.metric)
    sys.exit(1)

  run(ground_truth, args.partial_dir, partial_table, args.partial_template, args.output_file,
      pool_methods, step, args.max, sort_datasets, check_datasets, args.metric)


if __name__ == '__main__':

  import importlib
  path = os.path.join(os.path.dirname(sys.argv[0]), '..', 'etc', 'compute_meta_prediction.py')
  spec = importlib.util.spec_from_file_location('compute_meta_prediction', path)
  compute_meta_prediction = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(compute_meta_prediction)

  path = os.path.join(os.path.dirname(sys.argv[0]), '..', 'etc', 'compute_metrics.py')
  spec = importlib.util.spec_from_file_location('compute_metrics', path)
  compute_metrics = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(compute_metrics)

  main()
