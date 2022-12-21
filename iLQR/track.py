"""
Racing track.

Please contact the author(s) of this library if you have any questions.
Author:  Haimin Hu (haiminh@princeton.edu)
Reference: ECE346@Princeton (Zixu Zhang, Kai-Chieh Hsu, Duy P. Nguyen)
"""

import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from pyspline.pyCurve import Curve


class Track:

  def __init__(
      self, center_line: np.ndarray = None, width_left: float = None,
      width_right: float = None, loop: bool = True
  ):
    """
    Constructs a track with a fixed width.

    Args:
        center_line (np.ndarray, optional): 2D numpy array containing samples
            of track center line [[x1,x2,...], [y1,y2,...]]. Defaults to None.
        width_left (float, optional): width of the track on the left side.
            Defaults to None.
        width_right (float, optional): width of the track on the right side.
            Defaults to None.
        loop (bool, optional): flag indicating if the track has loop.
            Defaults to True.
    """
    self.width_left = width_left
    self.width_right = width_right
    self.loop = loop

    if center_line is not None:
      self.center_line = Curve(x=center_line[0, :], y=center_line[1, :], k=3)
      self.length = self.center_line.getLength()
    else:
      self.length = None
      self.center_line = None

    self.track_bound = None
    self.track_center = None

  def _interp_s(self, s: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of s (progress since start), returns corresponing (x,y) points
    on the track. In addition, return slope of trangent line on those points.

    Args:
        s (list): progress since start.

    Returns:
        np.ndarray: (x,y) points on the track.
        np.ndarray: slopes of trangent line.
    """
    try:
      n = len(s)
    except:
      s = np.array([s])
      n = len(s)

    interp_pt = self.center_line.getValue(s)
    slope = np.zeros(n)

    for i in range(n):
      deri = self.center_line.getDerivative(s[i])
      slope[i] = np.arctan2(deri[1], deri[0])
    return interp_pt.T, slope

  def interp(self, theta_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of theta (progress since start), return corresponing (x,y)
    points on the track. In addition, return slope of trangent line on those
    points.

    Args:
        theta_list (list): progress since start

    Returns:
        np.ndarray: (x,y) points on the track.
        np.ndarray: slopes of trangent line.
    """
    if self.loop:
      s = np.remainder(theta_list, self.length) / self.length
    else:
      s = np.array(theta_list) / self.length
      s[s > 1] = 1
    return self._interp_s(s)

  def get_closest_pts(
      self, points: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets the closest points.

    Args:
        points (np.ndarray): points on the track of [2xn] shape.

    Returns:
        np.ndarray: (x,y) points on the track.
        np.ndarray: slopes of trangent line.
        np.ndarray: projected progress
    """
    s, _ = self.center_line.projectPoint(points.T, eps=1e-3)
    closest_pt, slope = self._interp_s(s)
    return closest_pt, slope, s * self.length

  def project_point(self, point: np.ndarray) -> np.ndarray:
    """
    Projects a point.

    Args:
        point (np.ndarray): points on the track of [2xn] shape.

    Returns:
        np.ndarray: projected points.
    """
    s, _ = self.center_line.projectPoint(point, eps=1e-3)
    return s * self.length

  def get_track_width(self,
                      theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets the width of the track.

    Args:
        theta (np.ndarray): progress since start.

    Returns:
        np.ndarray: left width array.
        np.ndarray: right width array.
    """
    temp = np.ones_like(theta)
    return self.width_left * temp, self.width_right * temp

  def plot_track(self):
    """
    Plots the track.
    """
    N = 500
    if self.track_bound is None:
      theta_sample = np.linspace(0, 1, N, endpoint=False) * self.length
      interp_pt, slope = self.interp(theta_sample)

      if self.loop:
        self.track_bound = np.zeros((4, N + 1))
      else:
        self.track_bound = np.zeros((4, N))

      self.track_bound[
          0, :N] = interp_pt[0, :] - np.sin(slope) * self.width_left
      self.track_bound[
          1, :N] = interp_pt[1, :] + np.cos(slope) * self.width_left

      self.track_bound[
          2, :N] = interp_pt[0, :] + np.sin(slope) * self.width_right
      self.track_bound[
          3, :N] = interp_pt[1, :] - np.cos(slope) * self.width_right

      if self.loop:
        self.track_bound[:, -1] = self.track_bound[:, 0]

    plt.plot(self.track_bound[0, :], self.track_bound[1, :], 'k-')
    plt.plot(self.track_bound[2, :], self.track_bound[3, :], 'k-')

  def plot_track_center(self):
    """
    Plots the center of the track.
    """
    N = 500
    if self.track_center is None:
      theta_sample = np.linspace(0, 1, N, endpoint=False) * self.length
      interp_pt, slope = self.interp(theta_sample)
      self.track_center = interp_pt
      print(len(slope))

    plt.plot(self.track_center[0, :], self.track_center[1, :], 'r--')


if __name__ == '__main__':
  import csv
  track_file = 'outerloop_center_smooth.csv'
  x = []
  y = []
  with open(track_file, newline='') as f:
    spamreader = csv.reader(f, delimiter=',')
    for i, row in enumerate(spamreader):
      if i > 0:
        x.append(float(row[0]))
        y.append(float(row[1]))

  center_line = np.array([x, y])
  track = Track(
      center_line=center_line, width_left=0.3, width_right=0.3, loop=True
  )

  track.plot_track()
  track.plot_track_center()
  plt.show()
