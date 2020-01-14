import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
import numpy as np
import cv2

def bgr2rgb(img):
  # OpenCV's BGR to RGB
  rgb = np.copy(img)
  rgb[..., 0], rgb[..., 2] = img[..., 2], img[..., 0]
  return rgb

def check_do_save(func):
  def inner(self, *args, **kwargs):
    if self.do_save:
      func(self, *args, **kwargs)

  return inner

class Plotter(object):
  def __init__(self, plot=True, rows=0, cols=0, num_images=0, out_folder=None, out_filename=None):
    self.save_counter = 1
    self.plot_counter = 1
    self.do_plot = plot
    self.do_save = out_filename is not None
    self.out_filename = out_filename
    self.set_filepath(out_folder)

    if (rows + cols) == 0 and num_images > 0:
      # Auto-calculate the number of rows and cols for the figure
      self.rows = np.ceil(np.sqrt(num_images / 2.0))
      self.cols = np.ceil(num_images / self.rows)
    else:
      self.rows = rows
      self.cols = cols

  def set_filepath(self, folder):
    if folder is None:
      self.filepath = None
      return

    if not os.path.exists(folder):
      os.makedirs(folder)
    self.filepath = os.path.join(folder, 'frame{0:03d}.png')
    self.do_save = True

  @check_do_save
  def save(self, img, filename=None):
    if self.filepath:
      filename = self.filepath.format(self.save_counter)
      self.save_counter += 1
    elif filename is None:
      filename = self.out_filename

    mpimg.imsave(filename, bgr2rgb(img))
