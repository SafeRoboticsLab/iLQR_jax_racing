"""
Util functions.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ECE346@Princeton (Zixu Zhang, Kai-Chieh Hsu, Duy P. Nguyen)
"""

import yaml
import csv
import numpy as np

from .track import Track


class Struct:
  """
  Struct for managing parameters.
  """

  def __init__(self, data):
    for key, value in data.items():
      setattr(self, key, value)


def load_config(file_path):
  """
  Loads the config file.

  Args:
      file_path (string): path to the parameter file.

  Returns:
      Struct: parameters.
  """
  with open(file_path) as f:
    data = yaml.safe_load(f)
  config = Struct(data)
  return config


def load_track(config):
  """
  Loads the track.

  Args:
      config (Struct): problem parameters.

  Returns:
      Track: the track object.
  """
  x_center_line = []
  y_center_line = []
  with open(config.TRACK_FILE) as f:
    spamreader = csv.reader(f, delimiter=',')
    for i, row in enumerate(spamreader):
      if i > 0:
        x_center_line.append(float(row[0]))
        y_center_line.append(float(row[1]))

  center_line = np.array([x_center_line, y_center_line])
  track = Track(
      center_line=center_line, width_left=config.TRACK_WIDTH_L,
      width_right=config.TRACK_WIDTH_R, loop=config.LOOP
  )
  return track


def plot_ellipsoids(
    ax, E_list, arg_list=[], dims=None, N=200, plot_center=True,
    use_alpha=False
):
  """
  Plot a list of ellipsoids
  Args:
      E_list (list): list of ellipsoids
      arg_list (list): list of args dictionary.
      dims (list): dimension to be preserved.
      N (int): number of boundary points.
      plot_center (bool): plot the center of the ellipsoid if True. Defaults to
          True.
      use_alpha (bool): make later ellipsoids more transparent if True.
          Defaults to False.
  """
  if len(arg_list) == 0:
    arg_list = [dict(c='r')] * len(E_list)
  elif len(arg_list) == 1:
    arg_list = arg_list * len(E_list)
  else:
    assert len(arg_list) == len(E_list), "The length does not match."
  if use_alpha:
    alpha_list = np.linspace(1., 0.1, len(E_list))
  else:
    alpha_list = np.ones(len(E_list))

  for E0, arg, alpha in zip(E_list, arg_list, alpha_list):
    if dims:
      E = E0.projection(dims)
    else:
      E = E0.copy()
    if E.dim() > 3:
      raise ValueError("[ellReach-plot] ellipsoid dimension can be 1, 2 or 3.")
    if E.dim() == 1:
      x = np.array([(E.q - np.sqrt(E.Q))[0, 0], (E.q + np.sqrt(E.Q))[0, 0]])
      ax.plot(x, np.zeros_like(x), color=arg)
      ax.plot(E.q, 0, color=arg, marker='*')
    if E.dim() == 2:
      if E.is_degenerate():
        print("degenerate!")
      phi = np.array(np.linspace(0., 2. * np.pi, N))
      L = np.concatenate(
          (np.cos(phi).reshape(1, N), np.sin(phi).reshape(1, N)), axis=0
      )
      _, x = E.rho(L)
      # x = np.concatenate((x, x[:, 0].reshape(2, 1)), axis=1)
      ax.plot(x[0, :], x[1, :], alpha=alpha, **arg)
      if plot_center:
        ax.plot(E.q[0, 0], E.q[1, 0], marker='.', alpha=alpha, **arg)
      ax.set_aspect('equal')
    if E.dim() == 3:
      raise NotImplementedError
