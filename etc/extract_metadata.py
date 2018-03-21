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
import csv
import json
import glob
import sys

version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)

# Assumes we are at the root ./ of the following tree
# ./duplicates-map.csv
# ./diagnoses-thesaurus.txt
# ./locations-thesaurus.txt
# ./edinburghDermofit
# ./edinburghDermofit/originalImages/
# ./edinburghDermofit/originalImages/lesionlist.txt
# ./edinburghDermofit/originalMasks/
# ./edraAtlas/
# ./edraAtlas/edraAtlasCelebiMetadata.csv
# ./edraAtlas/allimages (all filenames in lowercase)
# ./edraAtlas/edra-metadata.csv
# ./isicArchive/images/
# ./isicArchive/meta/
# ./isicArchive/masksImages/
# ./isicArchive/masksMeta/
# ./isicChallenge2017/
# ./isicChallenge2017/ISIC-2017_Training_Data/
# ./isicChallenge2017/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv
# ./isicChallenge2017/ISIC-2017_Validation_Data/
# ./isicChallenge2017/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data_metadata.csv
# ./isicChallenge2017/ISIC-2017_Test_v2_Data/
# ./isicChallenge2017/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data_metadata.csv
# ./isicChallenge2017/ISIC-2017_Training_Part1_GroundTruth/
# ./isicChallenge2017/ISIC-2017_Validation_Part1_GroundTruth/
# ./isicChallenge2017/ISIC-2017_Test_v2_Part1_GroundTruth/
# ./isicChallenge2017/ISIC-2017_Training_Part3_GroundTruth.csv
# ./isicChallenge2017/ISIC-2017_Validation_Part3_GroundTruth.csv
# ./isicChallenge2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv
# ./portoPH2/
# ./portoPH2/portoPH2Metadata.csv
# ./portoPH2/PH2\ Dataset\ images/*


# =====================================
# Loads and parses the duplicates' map
duplicates = [ row.strip().split(';') for row in open('duplicates-map.csv') if row.strip()!='' ]
duplicates = { d[0] : d[1] for d in duplicates }


# =====================================
# Loads and parses the diagnoses thesaurus
diagnoses = [ row.strip().split('\t') for row in open('diagnoses-thesaurus.txt')
        if row.strip()!='' ]
diagnoses = { d[1] : d[0] for d in diagnoses }
# diagnoses['diagnosis'] = ''

# Loads and parses the locations thesaurus
locations_list = [ row.strip().split('\t') for row in open('locations-thesaurus.txt') ]
locations = { '' : '', 'unknown' : '' }
location_target = None
for loc in locations_list:
  if location_target is None:
    location_target = loc[0].strip()
    assert location_target!=''
  elif loc[0].strip()=='':
    location_target = None
  else:
    locations[loc[1].strip()] = location_target


# =====================================
# Creates the empty metadata file
output_meta = open('allimages-metadata.csv', 'wt')
print('dataset;split;image;image_type;diagnosis;diagnosis_method;diagnosis_difficulty;'
    'diagnosis_confidence;lesion_thickness;lesion_diameter;lesion_location;age;sex;case;'
    'alias;semiduplicate', file=output_meta)

def check(value):
  if value is None:
    return ''
  elif not isinstance(value, str):
    return value
  elif value=='unknown':
    return ''
  else:
    return value.strip()

def try_to_float(value):
  if isinstance(value, float):
    return str(value)
  if isinstance(value, int):
    return str(float(value))
  try :
    return str(float(str(value).replace(',', '.')))
  except (AttributeError, ValueError):
    return ''

def try_to_int(value):
  if isinstance(value, int):
    return str(value)
  try :
    return str(int(str(value)))
  except (AttributeError, ValueError):
    return ''

def print_lesion(dataset='', split='', image=None, image_type='', diagnosis='', diagnosis_method='',
    diagnosis_difficulty='', diagnosis_confidence='', lesion_thickness='', lesion_diameter='',
    lesion_location='', age='', sex='', case='', alias=''):
  print(check(dataset), check(split), check(image), check(image_type), check(diagnosis),
      check(diagnosis_method), check(diagnosis_difficulty), check(diagnosis_confidence),
      check(lesion_thickness), check(lesion_diameter), check(lesion_location), check(age),
      check(sex), check(case), check(alias),
      duplicates.get(image, ''), sep=';', file=output_meta)


# =====================================
# Edinburgh Dermofit Dataset - processes and creates metadata
#
# Ballerini, L., Fisher, R. B., Aldridge, R. B., Rees, J. (2013). A Color and Texture Based
# Hierarchical K-NN Approach to the Classification of Non-melanoma Skin Lesions, Color Medical Image
# Analysis. Lecture Notes in Computational Vision and Biomechanics 6 (M. E. Celebi, G. Schaefer, eds.)
#
# 5.1 Acquisition and preprocessing
# Our image database comprises 960 lesions, belonging to 5 classes (45 AK, 239 BCC, 331 ML, 88 SCC,
# 257 SK). The ground truth used for the experiments is based on the agreed classifications by 2
# dermatologists and a pathologist. Images are acquired using a Canon EOS 350D SLR camera. Lighting
# was controlled using a ring flash and all images were captured at the same distance (∼50 cm)
# resulting in a pixel resolution of about 0.03 mm. Lesions are segmented using the region-based
# active contour approach described in [39]. The segmentation method uses a statistical model based
# the level-set framework. Morphological opening has been applied to the segmented lesions to be
# sure to have patches containing only lesions and healthy skin where the features are extracted.
#
# https://licensing.eri.ed.ac.uk/i/software/dermofit-image-library.html

def process_edinburgh():
  edinburgh_diagnoses = {
    'AK'   : 'Actinic Keratosis',
    'BCC'  : 'Basal Cell Carcinoma',
    'DF'   : 'Dermatofibroma',
    'IEC'  : 'Intraepithelial Carcinoma',
    'MEL'  : 'Malignant Melanoma',
    'ML'   : 'Melanocytic Nevus (mole)',
    'PYO'  : 'Pyogenic Granuloma',
    'SCC'  : 'Squamous Cell Carcinoma',
    'SK'   : 'Seborrhoeic Keratosis',
    'VASC' : 'Haemangioma',
  }

  edinburgh_lesions = open('./edinburghDermofit/originalImages/lesionlist.txt', 'rt')
  edinburgh_lesions = [ row.strip().split() for row in edinburgh_lesions if row.strip()!='' ]

  for lesion in edinburgh_lesions:
    print_lesion(dataset='edinburghDermofit', image=lesion[1].strip(), image_type='clinical',
        diagnosis=diagnoses[edinburgh_diagnoses[lesion[2].strip()]],
        diagnosis_method='histopathology', diagnosis_confidence='high')


# =====================================
# EDRA Atlas of Dermoscopy Dataset - processes and creates metadata
#
# http://www.dermoscopy.org/atlas/autori_affiliati.asp

def process_edra():
  edra_methods = {
    'clinical follow up'     : 'follow-up',
    'excision'               : 'histopathology',
    'no further examination' : 'clinic',
  }
  edra_thickness = {
    'Melanoma (less than 0.76 mm)'  : '<0.76',
    'Melanoma (0.76 - 1.5 mm)'      : '0.76-1.5',
    'Melanoma (more than 1.5 mm)'   : '>1.5',
  }

  edra_ids = { '' : True }

  edra_metadata = csv.reader(open('./edraAtlas/edraAtlasCelebiMetadata.csv', newline=''),
      delimiter=';', quotechar='"')
  fields = next(edra_metadata)
  assert fields==['Dermatoscopic', 'Clinical', 'Case_Number', 'Diagnosis', 'Location', 'Age', 'Sex',
      'Diameter', 'Elevation', 'Diagnostic_Difficulty', 'Management', 'Global_feature',
      'Pigment_net', 'Dots_and_globules', 'Streaks', 'BW_Veil', 'Pigmentation',
      'Hypopigmentation', 'Regression_structures', 'Vascular_structures', 'Other_criteria']
  for lesion in edra_metadata:
    diagnosis = diagnoses[lesion[3]]
    thickness = edra_thickness.get(lesion[3], '')
    diagnosis_method = edra_methods[lesion[10]]
    location = locations[lesion[4]]
    if not lesion[0] in edra_ids:
      print_lesion(dataset='edraAtlas', image=lesion[0], image_type='dermoscopic',
        diagnosis=diagnosis, diagnosis_method=diagnosis_method, diagnosis_difficulty=lesion[9],
        diagnosis_confidence='high', lesion_thickness=thickness, lesion_diameter=try_to_float(lesion[7]),
        lesion_location=location, age=try_to_int(lesion[5]), sex=lesion[6], case=lesion[2])
      edra_ids[lesion[0]] = True
    if not lesion[1] in edra_ids:
      print_lesion(dataset='edraAtlas', image=lesion[1], image_type='clinical',
        diagnosis=diagnosis, diagnosis_method=diagnosis_method, diagnosis_difficulty=lesion[9],
        diagnosis_confidence='high', lesion_thickness=thickness, lesion_diameter=try_to_float(lesion[7]),
        lesion_location=location, age=try_to_int(lesion[5]), sex=lesion[6], case=lesion[2])
      edra_ids[lesion[1]] = True


# =====================================
# ISIC Archive 2017 Datatset
#
# https://isic-archive.com/

def process_isic_archive():
  isic_methods = {
    '' : '',
    'histopathology' : 'histopathology',
    'single image expert consensus' : 'clinic',
  }

  isic_metadata = {}

  for metafile in glob.iglob('./isicArchive/meta/*.json'):
    meta = json.load(open(metafile, 'rt'))
    image = meta['_id'].strip()
    alias = meta['name'].strip()
    try :
      diagnosis = diagnoses[meta['meta']['clinical']['diagnosis'].strip()]
    except (KeyError, AttributeError):
      diagnosis = ''
    if meta['meta']['clinical']['diagnosis_confirm_type']:
      method = isic_methods[meta['meta']['clinical']['diagnosis_confirm_type']]
    else:
      method = ''
    confidence = 'high' if method=='histopathology' else 'low'
    try :
      diameter = try_to_float(meta['meta']['clinical']['clin_size_long_diam_mm'])
    except KeyError:
      diameter = ''
    unstructured = meta['meta']['unstructured']
    if 'Breslow' in unstructured:
      try :
        thickness = try_to_float(unstructured['Breslow'].split()[0])
      except (IndexError, KeyError, ValueError):
        thickness = ''
    elif 'mel_thick' in unstructured:
      try :
        thickness = try_to_float(unstructured['mel_thick'].split()[0])
      except (IndexError, KeyError, ValueError):
        thickness = ''
    else:
      thickness = ''
    possible_locations = ['Location', 'location', 'anatomic', 'anatom_site_general',
        'localization', 'quantloc']
    locs = set(( locations[unstructured[loc].strip()] for loc in possible_locations
         if loc in unstructured ))
    locs.discard('')
    n_locs = len(locs)
    if n_locs == 0 :
      locs = ''
    elif n_locs == 1 :
      locs = locs.pop()
    else:
      locs_orig = set(( unstructured[loc].strip() for loc in possible_locations
           if loc in unstructured ))
      locs_orig.discard('')
      print('Warning: %s has many locations %s (from: %s)' %
          (image, '/'.join(locs), '/'.join(locs_orig)), end=' ', file=sys.stderr)
      if n_locs == 2:
        locs = list(locs)
        # If two locations are compatible, prefers the most specific
        if locs[0] in locs[1]:
          locs = locs[1]
        elif locs[1] in locs[0]:
          locs = locs[0]
        # If two locations are incompatible, prefers the most general
        elif locs[0].count('.') <= locs[1].count('.'):
          print('*', end='', file=sys.stderr)
          locs = locs[0]
        else:
          print('*', end='', file=sys.stderr)
          locs = locs[1]
      else:
        # If there are multiplie locations, prefers the most common
        locs = max(locs, key=locs.count)
      print('=>',locs, file=sys.stderr)
    age = str(meta['meta']['clinical']['age_approx'])
    age = age if age!='None' else ''
    sex = meta['meta']['clinical']['sex']
    if 'patient' in unstructured:
      case = unstructured['patient'].strip()
    elif 'lesion_id' in unstructured:
      case = unstructured['lesion_id'].strip()
    elif 'lesion id' in unstructured:
      case = unstructured['lesion id'].strip()
    elif 'id1' in unstructured:
      case = unstructured['id1'].strip()
    else:
      case = ''
    print_lesion(dataset='isicArchive', image=image, image_type='dermoscopic',
      diagnosis=diagnosis, diagnosis_method=method, diagnosis_confidence=confidence,
      lesion_thickness=thickness, lesion_diameter=diameter, lesion_location=locs, age=age,
      sex=sex, case=case, alias=alias)
    isic_metadata[alias] = (image, diagnosis, method, confidence, thickness, diameter, locs,
      age, sex, case)

  return isic_metadata


# =====================================
# ISIC Challenge 2017 Datatset
#
# https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a

def process_isic_challenge_split(split, isic_metadata, metadata_filename, truth_filename) :
  challenge_metadata = csv.reader(open(metadata_filename, newline=''), delimiter=',', quotechar='"')
  fields = next(challenge_metadata)
  assert fields==['image_id', 'age_approximate', 'sex']
  challenge_truth = csv.reader(open(truth_filename, newline=''), delimiter=',', quotechar='"')
  fields = next(challenge_truth)
  assert fields==['image_id', 'melanoma', 'seborrheic_keratosis']
  challenge_truth = { t[0] : (int(float(t[1])), int(float(t[2]))) for t in challenge_truth }
  for lesion in challenge_metadata:
    if challenge_truth[lesion[0]] == (0, 0):
      diagnosis = diagnoses['Nevus']
    elif challenge_truth[lesion[0]] == (1, 0):
      diagnosis = diagnoses['Melanoma']
    elif challenge_truth[lesion[0]] == (0, 1):
      diagnosis = diagnoses['Seborrheic keratosis']
    else:
      assert False # Invalid diagnosis
    if lesion[0] in isic_metadata :
      # If the lesion is also on Archive, use metadata there instead
      meta = isic_metadata[lesion[0]]
      age = lesion[1]
      sex = lesion[2]
      age = age if age!='' else meta[7]
      sex = sex if sex!='' else meta[8]
      if not diagnosis in meta[1]:
        print("Error: %s diagnoses incompatible: %s vs. %s" %
            (lesion[0], diagnosis, meta[1]), file=sys.stderr)
      print_lesion(dataset='isicChallenge2017', split=split, image=lesion[0],
        image_type='dermoscopic', diagnosis=meta[1], diagnosis_method=meta[2],
        diagnosis_confidence=meta[3], lesion_thickness=meta[4], lesion_diameter=meta[5],
        lesion_location=meta[6], age=age, sex=sex, case=meta[9], alias=meta[0])
    else :
      print_lesion(dataset='isicChallenge2017', split=split, image=lesion[0],
        image_type='dermoscopic', diagnosis=diagnosis, age=lesion[1], sex=lesion[2])

def process_isic_challenge(isic_metadata):
  # ... training split
  process_isic_challenge_split('train', isic_metadata,
      './isicChallenge2017/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv',
      './isicChallenge2017/ISIC-2017_Training_Part3_GroundTruth.csv')
  # ... validation split
  process_isic_challenge_split('validation', isic_metadata,
      './isicChallenge2017/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data_metadata.csv',
      './isicChallenge2017/ISIC-2017_Validation_Part3_GroundTruth.csv')
  # ... test split
  process_isic_challenge_split('test', isic_metadata,
      './isicChallenge2017/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data_metadata.csv',
      './isicChallenge2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv')


# =====================================
# Porto PH2 Datatset
#
# The dermoscopic images were obtained at the Dermatology Service of Hospital Pedro Hispano
# (Matosinhos, Portugal) under the same conditions through Tuebinger Mole Analyzer system using a
# magnification of 20x. They are 8-bit RGB color images with a resolution of 768x560 pixels.
#
# This image database contains a total of 200 dermoscopic images of melanocytic lesions, including
# 80 common nevi, 80 atypical nevi, and 40 melanomas. The PH2 database includes medical annotation
# of all the images namely medical segmentation of the lesion, clinical and histological diagnosis
# and the assessment of several dermoscopic criteria (colors; pigment network; dots/globules;
# streaks; regression areas; blue-whitish veil).
# The assessment of each parameter was performed by an expert dermatologist, according to the
# following parameters: (...)
# https://www.fc.up.pt/addi/ph2%20database.html

def process_porto():
  porto_metadata = csv.reader(open('./portoPH2/portoPH2Metadata.csv', newline=''),
      delimiter=';', quotechar='"')
  fields = next(porto_metadata)
  assert fields==['Image_Name', 'Histological_Diagnosis', 'Common_Nevus', 'Atypical_Nevus',
      'Melanoma', 'Asymmetry', 'Pigment_Network', 'Dots_Globules', 'Streaks',
      'Regression_Areas', 'Blue-Whitish_Veil', 'White', 'Red', 'Light-Brown', 'Dark-Brown',
      'Blue-Gray', 'Black']
  for lesion in porto_metadata:
    if lesion[1]!='':
      diagnosis = diagnoses[lesion[1]]
    elif lesion[2]!='':
      diagnosis = diagnoses['Common Nevus']
    elif lesion[3]!='':
      diagnosis = diagnoses['Atypical Nevus']
    elif lesion[4]!='':
      diagnosis = diagnoses['Melanoma']
    else:
      diagnosis = ''
    print_lesion(dataset='portoPH2', image=lesion[0], image_type='dermoscopic',
      diagnosis=diagnosis, diagnosis_method='clinic')


# =====================================
# Processes all datasets
#
def main():
  process_edinburgh()
  process_edra()
  isic_metadata = process_isic_archive()
  process_isic_challenge(isic_metadata)
  process_porto()

if __name__ == '__main__':
  main()
