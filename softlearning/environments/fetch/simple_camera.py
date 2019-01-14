#Adapted from https://github.com/tensorflow/models/blob/master/research/tcn/dataset/webcam.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
from multiprocessing import Process
import os
import subprocess
import sys
import time
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import animation  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class SimpleCamera(object):

  def __init__(self,port):

    self.get_camera(port)
    self.current_img = None
    self.height = 100
    self.width = 100

  def get_camera(self,port):

    camera = cv2.VideoCapture(port)

    if not camera.isOpened():
      try:
        # Try to find and kill hanging cv2 process_ids.
        output = subprocess.check_output(['lsof -t /dev/video*'], shell=True)
        print('Found hanging cv2 process_ids: \n')
        print(output)
        print('Killing hanging processes...')
        for process_id in output.split('\n')[:-1]:
          subprocess.call(['kill %s' % process_id], shell=True)
        time.sleep(3)
        # Recapture webcam.
        camera = cv2.VideoCapture(port)
      except subprocess.CalledProcessError:
        raise ValueError(
              'Cannot connect to cameras.')

    # Verify camera is able to capture images.
    self.active_camera = camera


  def capture(self, timestep):

    im = self.get_image()

    return np.array(im)


  def close(self):
    if self.active_camera:
      self.active_camera.release()
      sys.exit(0)


  def get_image(self):
    for i in range(5):
      data = self.active_camera.read()
    _, im = data
    im = np.array(im)
    return im

def preprocess_image(np_image):

    np_image = np_image[...,[2,1,0]]
    image = Image.fromarray(np_image, 'RGB')

    #image = image.convert('L')
    #left, top, right, bottom
    image = image.crop((80, 0, 560, 480))
    image = image.resize((100, 100))

    return np.array(image)