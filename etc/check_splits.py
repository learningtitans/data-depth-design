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

import sys
import traceback

version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)


formula = ' '.join(sys.argv[1:]).split()

if len(formula)<3 or not formula[1] in ('contains', 'equals', 'intersects', 'intersection', 'diff', 'symdiff'):
  print('usage: check_splits.py csv_file:delimiter:field[:slice] [contains|equals|intersects|intersection|diff|symdiff] csv_file:delimiter:field[:slice]', file=sys.stderr)
  sys.exit(2)

def read_set(set_spec):
  set_spec = set_spec.split(':')
  file_name = set_spec[0]
  delimiter = set_spec[1]
  field = int(set_spec[2])
  elements = [ e.strip().split(delimiter)[field].strip() for e in open(file_name) ]
  if len(set_spec)>3:
    begin_slice= int(set_spec[3]) if set_spec[3].strip() else None
    end_slice = int(set_spec[4]) if len(set_spec)>4 and set_spec[3].strip() else None
    elements = elements[begin_slice:end_slice]
  return set(elements)

def main(formula):
  try:
    first_set = read_set(formula[0])
    operator = formula[1]
    second_set = read_set(formula[2])
  except Exception:
    problem = traceback.format_exc()
    print(problem, file=sys.stderr)
    return 2

  if operator=='equals':
    if first_set==second_set:
      print('True')
      return 1
    else:
      print('False')
      return 0
  elif operator=='contains':
    if first_set.issuperset(second_set):
      print('True')
      return 1
    else:
      print('False')
      return 0
  elif operator=='intersects':
    common = first_set & second_set
    if common:
      print('True')
      return 1
    else:
      print('False')
      return 0
  elif operator=='intersection':
    common = first_set & second_set
    if common:
      print('\n'.join(common))
      return 1
    else:
      return 0
  elif operator=='diff':
    diff = first_set - second_set
    if diff:
      print('\n'.join(diff))
      return 1
    else:
      return 0
  elif operator=='symdiff':
    diff = first_set ^ second_set
    if diff:
      print('\n'.join(diff))
      return 1
    else:
      return 0
  else:
    assert False

if __name__ == '__main__':
    sys.exit(main(formula))

