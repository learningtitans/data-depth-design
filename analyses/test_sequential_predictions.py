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
import os.path
import random
import sys

import pandas as pd

version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test_all_meta_predictions')

def run(partial_table, output_file, sort_dataset, check_dataset,
    metric='isbi_auc', seed=0, n_samples=100):

  factors = ( 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', )
  n_factors = len(factors)

  random.seed(a=seed)
  print('sort_dataset', 'check_dataset', 'sequence', 'kickoff', metric, sep=',', file=output_file)

  for exp in range(n_samples):

    sequence = list(factors)
    random.shuffle(sequence)

    sort_data  = partial_table[partial_table["j"]==sort_dataset]
    check_data = partial_table[partial_table["j"]==check_dataset]

    # Starts from a randomly chosen point
    kickoff = tuple(( -1 if random.random()<0.5 else 1 for n in range(n_factors) ))
    hyper = { f : h for f,h in zip(factors, kickoff) }

    def to_query(hyper):
      return ' and '.join(( '%s==%s' % (k,v) for k,v in hyper.items() ))

    # Optimizes sequentially
    for h in sequence:
      try :
        hyper[h] = -1
        value_minus = sort_data.query(to_query(hyper)).iloc[0].loc[metric]
      except IndexError:
        log.error("ERROR: the treatment %s was not found", to_query(hyper))
        value_minus = float('-inf')
      try :
        hyper[h] = 1
        value_plus = sort_data.query(to_query(hyper)).iloc[0].loc[metric]
      except IndexError:
        log.error("ERROR: the treatment %s was not found", to_query(hyper))
        value_plus = float('-inf')
      if value_minus >= value_plus:
        # We slightly favor value_minus because, by convention, it's
        # usually the cheaper option
        hyper[h] = -1

    # Checks which performance was actually obtained
    final_perf = check_data.query(to_query(hyper)).iloc[0].loc[metric]
    print(sort_dataset, check_dataset, ''.join(sequence), '"' + ' '.join((str(k) for k in kickoff)) + '"', final_perf, sep=',', file=output_file)
    print(sort_dataset, check_dataset, ''.join(sequence), '"' + ' '.join((str(k) for k in kickoff)) + '"', final_perf, sep=',')


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--partial_table', required=True,
      help='File with the metrics for the partial predictions.')
  parser.add_argument('--output_file', type=argparse.FileType('w'), required=True,
     help='Output file with the performance of the differente sequential predictions.')
  parser.add_argument('--sort_dataset', type=int,
      help='Test dataset used to sequentially decide the hyperparameters.')
  parser.add_argument('--check_dataset', type=int,
      help='Test dataset used to measure the performance obtained.')
  parser.add_argument('--metric', type=str, default='isbi_auc',
      help='Metric used to sort the techniques and measure the meta-prediction performance.')
  parser.add_argument('--n_samples', type=int, default=100,
      help='Number of sequential experiments to perform, defaults to 100.')
  parser.add_argument('--random_seed', type=int, default=0,
      help='Seed used to sample experiments, defaults to 0.')
  parser.add_argument('--verbose',
      help='increase output verbosity', action='store_true')
  parser.add_argument('--allow_old_python',
      help='The script was not tested on Python 2 and will normally require Python 3.4+, '
        'but this flag allows using older versions (use it at your own risk).', action='store_true')
  args = parser.parse_args()

  if args.verbose:
    log.setLevel(logging.DEBUG)

  partial_table = pd.read_csv(args.partial_table, sep=";") # TODO: dtype

  sort_dataset  = args.sort_dataset
  if sort_dataset<0:
    log.fatal("--sort_dataset must be a non-negative integer")
    sys.exit(1)

  check_dataset  = args.check_dataset
  if check_dataset<0:
    log.fatal("--check_dataset must be a non-negative integer")
    sys.exit(1)

  if args.metric not in partial_table.columns.values:
    log.fatal("--metric must be one f the headers of the partial_table")
    sys.exit(1)

  run(partial_table, args.output_file, sort_dataset, check_dataset,
     args.metric, args.random_seed, args.n_samples)


if __name__ == '__main__':

  import importlib
  path = os.path.join(os.path.dirname(sys.argv[0]), '..', 'etc', 'compute_metrics.py')
  spec = importlib.util.spec_from_file_location('compute_metrics', path)
  compute_metrics = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(compute_metrics)

  main()
