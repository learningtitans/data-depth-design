#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Contributed 2017 Julia Tavares
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import sys

version_required = (3, 4, 0)
version_running  = sys.version_info
if version_running<version_required and '--allow_old_python' not in sys.argv:
  print('This script requires Python %s or superior, your version: %s' %
      ('.'.join((str(v) for v in version_required)),
       '.'.join((str(v) for v in version_running )),), file=sys.stderr)
  sys.exit(1)


def resize_mask(filename, width, height, output_folder=""):
  if not os.path.exists(filename): # remove file extension and try again
    ext = filename.rsplit(".",1)
    if len(ext)>1:
      filename = ext[0]
    filename = filename + ".npy"
  tmp = np.load(filename)
  name = os.path.basename(filename).replace(".npy","")
  tmp = cv2.resize(tmp.reshape(tmp.shape[1], tmp.shape[2]), (width, height), interpolation = cv2.INTER_CUBIC)
  _, tmp = cv2.threshold(tmp,0.5,255,cv2.THRESH_BINARY)
  if output_folder:
    # print(os.path.join(output_folder, name + "_segmentation.png"))
    cv2.imwrite(os.path.join(output_folder, name + "_segmentation.png"), tmp)
  return tmp

size = int(sys.argv[1])
source = sys.argv[2]
dest = sys.argv[3]
print("Resizing images to %dx%d from %s to %s" % (size, size, source, dest))

width, height = size, size
output_folder = dest
for fname in tqdm(glob.glob(os.path.join(source, "*"))):
  resized_img = resize_mask(fname, width, height, output_folder=output_folder)
  assert resized_img.shape==(width, height)
