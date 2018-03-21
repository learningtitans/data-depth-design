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
import collections
import logging
import random
import sys

version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)


# Seed for repeatability.
_RANDOM_SEED = 8191

# Command-line arguments and logging
def percentage(value):
  try:
    fvalue = float(value)
  except ValueError:
    fvalue = float('nan')
  if not (fvalue>=0.0 and fvalue<=100.0): # Deals with NaNs correctly
    raise argparse.ArgumentTypeError('invalid percentage %s - '
        'enter a value between 0.0 and 100.0.' % value)
  return fvalue

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', type=argparse.FileType('r'), required=True,
    help='File with the dataset metadata.')
parser.add_argument('--trainoutput', type=argparse.FileType('w'),
    help='File to receive the output of the training split (required unless --train_perc=0).')
parser.add_argument('--testoutput', type=argparse.FileType('w'),
    help='File to receive the output of the test split (required unless --train_perc=100).')
parser.add_argument('--blacklist', type=argparse.FileType('r'),
    help='File with the id of all blacklisted images, to be excluded from the dataset.')
parser.add_argument('--trainlist', type=argparse.FileType('r'),
    help='File with the id of all images to be forced to belong to the training split.')
parser.add_argument('--testlist', type=argparse.FileType('r'),
    help='File with the id of all images to be forced to belong to the test split.')
parser.add_argument('--train_perc', type=percentage, default=85.0,
    help='The fraction of images to include in the training split (in %).')
parser.add_argument('--verbose', help='increase output verbosity',
                    action='store_true')
parser.add_argument('--allow_old_python',
    help='The script was not tested on Python 2 and will normally require Python 3.4+, '
      'but this flag allows using older versions (use it at your own risk).', action='store_true')
args = parser.parse_args()

if args.verbose:
  logging.basicConfig(level=logging.DEBUG)
else:
  logging.basicConfig(level=logging.INFO)

log = logging.getLogger('create_splits')


def run(metadata_file, train_perc, train_output_file, test_output_file,
    blacklist_file=None, trainlist_file=None, testlist_file=None):

  # Get metadata
  _i_dataset              =  0
  _i_split                =  1
  _i_image                =  2
  _i_id                   =  _i_image
  _i_image_type           =  3
  _i_diagnosis            =  4
  _i_diagnosis_method     =  5
  _i_diagnosis_difficulty =  6
  _i_diagnosis_confidence =  7
  _i_lesion_thickness     =  8
  _i_lesion_diameter      =  9
  _i_lesion_location      = 10
  _i_age                  = 11
  _i_sex                  = 12
  _i_case                 = 13
  _i_alias                = 14
  _i_semiduplicate        = 15
  _i_assigned_case        = 16  # New field

  _DELIMITER = ';'
  metadata = [ m.strip().split(_DELIMITER) for m in metadata_file ]
  metadata = [ [ f.strip() for f in m] for m in metadata ]

  # Checks and skips header
  _EXPECTED_HEADER=('dataset;split;image;image_type;diagnosis;diagnosis_method;'
                    'diagnosis_difficulty;diagnosis_confidence;'
                    'lesion_thickness;lesion_diameter;lesion_location;age;sex;'
                    'case;alias;semiduplicate')
  _EXPECTED_FIELDS=_EXPECTED_HEADER.count(_DELIMITER)+1
  if _DELIMITER.join(metadata[0][:_EXPECTED_FIELDS])!=_EXPECTED_HEADER:
    log.fatal('invalid header on metadata file')
    sys.exit(1)
  metadata = metadata[1:]

  # Get blacklist
  if not blacklist_file is None:
    blacklist = set(( r.strip() for r in blacklist_file ))
  else:
    blacklist = set()
  # ... removes blacklisted images
  aliases = { m[_i_id] : m[_i_alias] for m in metadata }
  if blacklist:
    before_blacklist = len(metadata)
    metadata = [ m for m in metadata if not (m[_i_id] in blacklist) and
                   not (aliases[m[_i_id]] in blacklist) ]
    after_blacklist = len(metadata)
    log.warning('%d images directly blacklisted, %d effectively removed' %
        (len(blacklist), before_blacklist-after_blacklist,))

  # This class controls how the synonimy between entries in the metadata files is treated
  class CaseKey(object):
    def __str__(self):
      return ('{ id: ' + str(self.iid) + ', alias:' + str(self.alias) +
              ', case:' + str(self.case) + ', dup:' + str(self.semiduplicate) + ' }')
    def __hash__(self):
      return self.my_hash
    def __lt__(self, other):
      if self.case!=other.case:
        return self.case<other.case
      if self.semiduplicate!=other.semiduplicate:
        return self.semiduplicate<other.semiduplicate
      if self.iid!=other.iid:
        return self.iid<other.iid
      if self.alias!=other.alias:
        return self.alias<other.alias
    def __eq__(self, other):
      if self.case:
        return self.case == other.case
      elif self.semiduplicate:
        return self.semiduplicate == other.semiduplicate
      elif self.iid and self.alias:
        return ((self.iid == other.iid) or (self.iid == other.alias) or
                (self.alias == other.iid) or (self.alias == other.alias))
      else:
        return self.iid == other.iid
    def __ne__(self, other):
      return not self.__eq__(other)
    def __init__(self, iid, alias, case, semiduplicate):
      self.iid = iid
      self.alias = alias
      self.case = case
      self.semiduplicate = semiduplicate
      if case:
        my_hash = case.__hash__()
      elif semiduplicate:
        my_hash = semiduplicate.__hash__()
      elif iid and alias:
        my_hash = min(iid.__hash__(), alias.__hash__())
      else:
        my_hash = iid.__hash__()
      self.my_hash = my_hash

  # Solves aliases, cases, and semiduplicates
  # ...creates necessary maps and sets
  ids_to_indices = {}
  cases_to_indices = {}
  ids_to_cases = {}
  available_cases = set()
  for i,m in enumerate(metadata):
    if not m[_i_diagnosis][:4] in ( '1.1.', '1.5.', '3.2.' ):
      log.fatal('Only melanomas, keratoses, and nevi should be present in the dataset.')
      sys.exit(1)
    m.append( CaseKey(m[_i_id], m[_i_alias], m[_i_case], m[_i_semiduplicate]) )
    ids_to_indices[m[_i_id]] = i
    ids_to_cases[m[_i_id]] = m[_i_assigned_case]
    available_cases.add(m[_i_assigned_case])
    try:
      cases_to_indices[m[_i_assigned_case]].append(i)
    except KeyError:
      cases_to_indices[m[_i_assigned_case]] = [ i ]
  # ...checks that aliases are always assigned to the same case
  for m in metadata:
    try :
      if m[_i_alias]:
        m_alias = metadata[ids_to_indices[m[_i_alias]]]
      else:
        continue
    except KeyError:
      continue
    if m[_i_assigned_case]!=m_alias[_i_assigned_case]:
      log.info('image %s and its alias %s assigned to different cases (%s vs %s).' %
            (m[_i_id], m_alias[_i_id], str(m[_i_assigned_case]), str(m_alias[_i_assigned_case]),))

  # Get trainlist
  if not trainlist_file is None:
    trainlist = set(( ids_to_cases.get(r.strip(), None) for r in trainlist_file ))
  else:
    trainlist = set()

  # Get testlist
  if not testlist_file is None:
    testlist = set(( ids_to_cases.get(r.strip(), None) for r in testlist_file ))
  else:
    testlist = set()

  if trainlist.intersection(testlist):
    log.fatal('''There are cases listed in both --trainlist and --testlist.
                        tra & tes: %s''' %
              ', '.join(trainlist.intersection(testlist)))
    sys.exit(1)

  # ...starts by dividing training and test sets respecting lists
  n_total = len(available_cases)
  train_set = set(( c for c in available_cases if c in trainlist ))
  test_set = set(( c for c in available_cases if c in testlist ))
  available_cases = available_cases - train_set - test_set
  assert len(available_cases)+len(train_set)+len(test_set) == n_total
  log.info('Preinit size of the splits - train: %d, test: %d' % (len(train_set), len(test_set),))

  # ...assigns a diagnosis to the remaining cases (for stratification purposes only)
  available_complete_cases = set()
  for c in available_cases:
    diagnoses = [ metadata[i][_i_diagnosis][:4] for i in cases_to_indices[c] ]
    diagnosis = collections.Counter(diagnoses).most_common(1)[0][0]
    available_complete_cases.add((c, diagnosis,))
  assert len(available_complete_cases) == len(available_cases)

  # ...divides the remaining cases in a stratified way
  available_complete_cases = sorted(list(available_complete_cases)) # Needed for determinism
  available_melanoma  = [ c[0] for c in available_complete_cases if c[1]=='3.2.' ]
  available_keratosis = [ c[0] for c in available_complete_cases if c[1]=='1.5.' ]
  available_nevus     = [ c[0] for c in available_complete_cases if c[1]=='1.1.' ]
  assert len(available_melanoma)+len(available_keratosis)+len(available_nevus) == len(available_cases)

  # Divide into train, and test sets...
  random.seed(_RANDOM_SEED)
  random.shuffle(available_melanoma)
  random.shuffle(available_keratosis)
  random.shuffle(available_nevus)

  # print('C', len(available_melanoma),len(available_keratosis),len(available_nevus))

  n_expectedTraining  = int(n_total*train_perc/100.0)
  n_currentTraining   = len(train_set)
  n_missingTraining   = max(0, n_expectedTraining-n_currentTraining)
  train_new_fraction  = n_missingTraining/n_total
  n_missing_melanoma  = int(len(available_melanoma) *train_new_fraction)
  n_missing_keratosis = int(len(available_keratosis)*train_new_fraction)
  n_missing_nevus     = int(len(available_nevus)    *train_new_fraction)

  # print('D1', n_expectedTraining, n_total, train_perc, n_currentTraining, n_missingTraining)
  # print('DM', len(available_melanoma), n_missing_melanoma)
  # print('DK', len(available_keratosis), n_missing_keratosis)
  # print('DN', len(available_nevus), n_missing_nevus)

  # print('D2', len(train_set),len(test_set), len(train_set)+len(test_set), n_total)
  train_set.update(available_melanoma [:n_missing_melanoma])
  train_set.update(available_keratosis[:n_missing_keratosis])
  train_set.update(available_nevus    [:n_missing_nevus])

  test_set.update(available_melanoma [n_missing_melanoma:])
  test_set.update(available_keratosis[n_missing_keratosis:])
  test_set.update(available_nevus    [n_missing_nevus:])

  # print('D3', len(train_set),len(test_set), len(train_set)+len(test_set), n_total)

  # Case contamination check
  trivial = set([''])
  if train_set.intersection(test_set) - trivial:
    log.fatal('''case contamination among sets.
                        tra & tes: %s''' %
         ', '.join(( str(c) for c in train_set.intersection(test_set) - trivial )))
    sys.exit(1)

  # Metadata list creation and check
  train_meta = [ m for m in metadata if m[_i_assigned_case] in train_set ]
  test_meta  = [ m for m in metadata if m[_i_assigned_case] in test_set  ]

  # print('E', len(train_set),len(test_set),n_total)

  # Code sanity checks
  assert len(train_meta) >= len(train_set)
  assert len(test_meta) >= len(test_set)
  assert len(train_set)+len(test_set) == n_total
  assert len(train_meta)+len(test_meta) == len(metadata)

  contamination = set(( m[_i_id] for m in train_meta)).intersection(
                      set(( m[_i_id] for m in test_meta))) - trivial
  if contamination:
    log.fatal('''ids contamination among sets.
                        tra & tes: %s''' %
              ', '.join(contamination))
    sys.exit(1)

  # Outputs splits
  if train_meta:
    print(_EXPECTED_HEADER, file=train_output_file)
    for m in train_meta:
      print(_DELIMITER.join(m[:_EXPECTED_FIELDS]), file=train_output_file)

  if test_meta:
    print(_EXPECTED_HEADER, file=test_output_file)
    for m in test_meta:
      print(_DELIMITER.join(m[:_EXPECTED_FIELDS]), file=test_output_file)

def main():
  if (args.train_perc!=0 or args.trainlist) and not args.trainoutput:
    log.fatal('The output file --trainoutput is mandatory if --train_perc>0 '
        'or --trainlist is present')
    sys.exit(1)
  if (args.train_perc!=100 or args.testlist) and not args.testoutput:
    log.fatal('The output file --testoutput is mandatory if --train_perc<100 '
        'or  --testlist is present')
    sys.exit(1)

  run(args.metadata, args.train_perc, args.trainoutput, args.testoutput,
      args.blacklist, args.trainlist, args.testlist)

if __name__ == '__main__':
  main()
