"""
Ellipsoid object for collision avoidance.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference:
    - ECE346@Princeton (Zixu Zhang, Kai-Chieh Hsu, Duy P. Nguyen)
    - Ellipsoidal Toolbox (Alex Kurzhanskiy)
    - ellReach (Python) Toolbox (Haimin Hu)
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np
import matplotlib


class EllipsoidObj():

  def __init__(
      self, q: np.ndarray = np.array([]), Q: np.ndarray = np.array([[], []]),
      r=None, n_circ=None, center_L=None, major_axis=None
  ):

    def symmetricize(Q):
      if Q.shape[1] == 0:
        return Q
      else:
        return (Q + Q.T) / 2

    self.q = q
    self.Q = symmetricize(Q)
    self.Qinv = np.linalg.inv(self.Q)

    if q.shape[0] != 2:
      raise ValueError(
          "The dimension should be 2 but get {}!".format(q.shape[0])
      )

    # Assume 2d ellipsoid。
    if r is None:
      # print("use eigval")
      eigVal, eigVec = np.linalg.eig(Q)
      eigVal = np.sqrt(eigVal)
      if eigVal[0] > eigVal[1]:
        a = eigVal[0]
        b = eigVal[1]
        self.major_axis = eigVec[:, 0]
      else:
        a = eigVal[1]
        b = eigVal[0]
        self.major_axis = eigVec[:, 1]
      self.r = b  # semi-minor axis
      self.n_circ = int(np.ceil(a / b))  #circles to cover the ellipsoid.
      # Assume local coordinate's origin is at the center of ellipsoid,
      # and x is align with major axis. Locations of circle centers are on
      # the local coordinate.
      self.center_L = np.linspace(-a + self.r, a - self.r, self.n_circ)
    else:
      self.r = r
      self.n_circ = n_circ
      self.center_L = center_L
      self.major_axis = major_axis

    # position of circle center on global coordinates
    self.center = np.einsum('n,a->an', self.center_L, self.major_axis) + self.q

    self.closest_pt = None
    self.slope = None
    self.width_L = None
    self.width_R = None

  def copy(self) -> EllipsoidObj:
    """
    Creats an identical ellipsoid object.

    Returns:
        EllipsoidObj: an ellipsoid object.
    """
    return EllipsoidObj(q=self.q, Q=self.Q)

  def is_degenerate(self) -> bool:
    """
    Checks if the ellipsoid is degenerate.
    Returns True if the ellipsoid is degenerate, False - otherwise.

    Returns:
        bool: flag indicating if the ellipsoid is degenerate.
    """
    return self.rank() < self.dim()

  def is_empty(self) -> bool:
    """
    Checks if the ellipsoid object is empty.
    Returns True if the ellipsoid is empty, False - otherwise.

    Returns:
        bool: flag indicating if the ellipsoid is empty.
    """
    return self.dim() == 0

  def dim(self) -> int:
    """
    Returns the dimension of the ellipsoid.

    Returns:
        int: dimension of the ellipsoid.
    """
    return self.q.size

  def rank(self) -> int:
    """
    Returns the rank of the shape matrix.

    Returns:
        int: rank of the shape matrix.
    """
    if self.dim() == 0:
      return 0
    else:
      return np.linalg.matrix_rank(self.Q)

  def parameters(self) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns parameters of the ellipsoid.

    Returns:
        np.ndarray: center of the ellipsoid.
        np.ndarray: shape matrix of the ellipsoid.
    """
    return self.q, self.Q

  def rho(self, L: np.ndarray,
          eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the support function of the ellipsoid in directions specified by
    the columns of matrix L, and boundary points X of this ellipsoid that
    correspond to directions in L.

    Args:
        L (np.ndarray): a direction matrix.
        eps (float): tolerance number.

    Returns:
        res (np.ndarray): the values of the support function for the specified
            ellipsoid and directions L.
        x (np.ndarray): boundary points of the ellipsoid that correspond to
            directions in L.
    """
    d = L.shape[1]
    res = np.array([]).reshape((1, 0))
    x = np.array([]).reshape((self.dim(), 0))
    for i in range(d):
      l = L[:, i].reshape(self.dim(), 1)
      sr = np.maximum(np.sqrt(l.T @ self.Q @ l)[0, 0], eps)
      res = np.hstack((res, self.q.T @ l + sr))
      x = np.hstack((x, self.Q @ l / sr + self.q))
    return res, x

  def __matmul__(self, A: np.ndarray) -> EllipsoidObj:
    """
    Multiplies the ellipsoid with a matrix or a scalar.
    If E(q,Q) is an ellipsoid, and A - matrix of suitable dimensions, then
        E(q, Q) @ A = E(Aq, AQA').

    Args:
        A (np.ndarray): a matrix.

    Returns:
        EllipsoidObj: an ellipsoid object.
    """
    q = A @ self.q
    Q = A @ self.Q @ A.T
    new_major_axis = A @ self.major_axis
    return EllipsoidObj(
        q=q, Q=Q, r=self.r, n_circ=self.n_circ, center_L=self.center_L,
        major_axis=new_major_axis
    )

  def __add__(self, b: np.ndarray) -> EllipsoidObj:
    """
    Computes vector addition with an ellipsoid:
    E + b = E(q + b, Q).

    Args:
        b (np.ndarray): a vector.
    """
    if b.shape != self.q.shape:
      raise ValueError("Dimensions of b and q of the ellipsoid do not match.")
    return EllipsoidObj(
        q=self.q + b, Q=self.Q, r=self.r, n_circ=self.n_circ,
        center_L=self.center_L, major_axis=self.major_axis
    )

  def add(self, b: np.ndarray):
    """
    Computes vector addition with an ellipsoid:
        E + b = E(q + b, Q).

    Args:
        b (np.ndarray): a vector.
    """
    self.q = self.q + b
    self.center = self.center + b

  def __sub__(self, b: np.ndarray) -> EllipsoidObj:
    """
    Computes vector subtraction from an ellipsoid:
        E - b = E(q - b, Q).

    Args:
        b (np.ndarray): a vector.
    """
    if b.shape != self.q.shape:
      raise ValueError("Dimensions of b and q of the ellipsoid do not match.")
    return EllipsoidObj(
        q=self.q - b, Q=self.Q, r=self.r, n_circ=self.n_circ,
        center_L=self.center_L, major_axis=self.major_axis
    )

  def plot_circ(self, ax: matplotlib.axes.Axes):
    """
    Plots a circle.

    Args:
        ax (matplotlib.axes.Axes): matplotlib axe object.
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    x0 = self.r * np.cos(theta)
    y0 = self.r * np.sin(theta)
    for i in range(self.n_circ):
      x = self.center[0, i] + x0
      y = self.center[1, i] + y0
      ax.plot(x, y)

  def projection(self, dims: List) -> EllipsoidObj:
    """
    Computes projection of the ellipsoid onto the given dimensions.

    Args:
        dims (List): dimension to be preserved.

    Returns:
        EllipsoidObj: projected ellipsoid (of lower dimension).
    """
    if not isinstance(dims, list):
      raise ValueError("Dims must be a list.")
    n = self.dim()
    m = len(dims)
    B = np.zeros((n, m))
    for i in range(m):
      B[dims[i], i] = 1
    return self @ B.T

  def is_internal(
      self, x: np.ndarray, abs_tol: float = 1e-7, slack: float = 0.
  ) -> bool:
    """
    Checks if the given point belongs to the ellipsoid, i.e. if inequality
        <(x - q), Q^(-1)(x - q)> <= 1
    holds.

    Args:
        x (np.ndarray): a column vector with the same dimension as the
            ellipsoid object.
        abs_tol (float, optional): tolerance number. Defaults to 1e-7.
        slack (float, optional): slack variable. Defaults to 0..

    Returns:
        bool: flag indicating if the given point belongs to the ellipsoid.
    """
    q = x - self.q
    r = (q.T @ self.Qinv @ q)[0, 0]
    if r < 1 + slack or np.abs(r - 1 + slack) < abs_tol:
      return True
    else:
      return False

  def plot(
      self, ax: matplotlib.axes.Axes, color: str = 'r', dims: List = None,
      N: int = 200, plot_center: bool = True
  ):
    """
    Plots the ellipsoid object in 1D, 2D or 3D.

    Args:
        ax (matplotlib.axes.Axes): matplotlib axe object.
        color (str): color scheme.
        dims (List): dimension to be preserved.
        N (int): number of boundary points.
        plot_center (bool): flag indicating if to plot the ellipsoid center.
    """
    if dims:
      E = self.projection(dims)
    else:
      E = self.copy()
    if E.dim() > 3:
      raise ValueError("Ellipsoid dimension can be 1, 2 or 3.")
    if E.dim() == 1:
      x = np.array([(E.q - np.sqrt(E.Q))[0, 0], (E.q + np.sqrt(E.Q))[0, 0]])
      ax.plot(x, np.zeros_like(x), color)
      ax.plot(E.q, 0, color + '*')
    if E.dim() == 2:
      phi = np.array(np.linspace(0., 2. * np.pi, N))
      L = np.concatenate(
          (np.cos(phi).reshape(1, N), np.sin(phi).reshape(1, N)), axis=0
      )
      _, x = E.rho(L)
      ax.plot(x[0, :], x[1, :], color)
      if plot_center:
        ax.plot(E.q[0, 0], E.q[1, 0], color + '.')
      ax.set_aspect('equal')
    if E.dim() == 3:
      raise NotImplementedError

  def obstacle_cost(self, obs: EllipsoidObj, q1: float, q2: float) -> float:
    """
    Computes the barrier cost given an obstacle and hyperparameters.

    Args:
        obs (EllipsoidObj): obstacle.
        q1 (float): shape parameter.
        q2 (float): scale parameter.

    Returns:
        float: barrier cost.
    """
    # Broadcasts to (2, self.n_circ, obs.n_circ).
    d = (self.center.T[:, np.newaxis, :] - obs.center.T).T
    # Reshapes to (2, self.n_circ*obs.n_circ).
    d = np.reshape(d, (2, self.n_circ * obs.n_circ), "F")
    # Ignores and sets to -0.2 when self is far away from the obstacle.
    g = np.clip(self.r + obs.r - np.linalg.norm(d, axis=0), -0.2, None)
    b = np.sum(q1 * np.exp(q2 * g))  # barrier
    return b

  def obstacle_derivative(
      self, state: np.ndarray, center_L: float, obs: EllipsoidObj, q1: float,
      q2: float
  ) -> Tuple(np.ndarray, np.ndarray):
    """
    Computes ellipsoidal obstacle soft constraint cost Jacobian and Hessian.

    Args:
        state (np.ndarray): nominal state trajectory, of the shape
            (x_dim, N).
        center_L (float): displacement of vehicle origin along the major
            axis of ellipsoid.
        obs (EllipsoidObj): obstacle (or its FRS) represented by ellipsoid.
        q1 (float): barrier function parameters.
        q2 (float): barrier function parameters.

    Returns:
        np.ndarray: Jacobian of softconstraint of shape (x_dim, N).
        np.ndarray: Hessian of softconstraint of shape (x_dim, x_dim, N).
    """
    x = state[0]
    y = state[1]
    theta = state[3]
    ct = np.cos(theta)
    st = np.sin(theta)
    center_L = self.center_L + center_L

    c_x = np.zeros(4)
    c_xx = np.zeros((4, 4))

    for i in range(self.n_circ):
      c_x_temp = np.zeros((4, obs.n_circ))
      c_xx_temp = np.zeros((4, 4, obs.n_circ))
      center_i = self.center[:, i][:, np.newaxis]
      dxy = center_i - obs.center
      dx = dxy[0, :]
      dy = dxy[1, :]
      d = np.linalg.norm(dxy, axis=0)

      b = self.r + obs.r - d

      idx_ignore = b < -0.2
      c = q1 * np.exp(q2 * b)

      # Jacobian
      c_x_temp[:2, :] = (-(q2 * c * dxy) / d)

      c_x_temp[3, :] = -(
          center_L[i] * q2 * c *
          (y*ct - obs.center[1, :] * ct - x*st + obs.center[0, :] * st)
      ) / d

      # Hessian
      c_xx_temp[0, 0, :] = ((q2 * c * dx**2) / (d**3) - (q2*c) / d +
                            (q2**2 * c * dx**2) / (d**2))
      c_xx_temp[1, 1, :] = ((q2 * c * dy**2) / (d**3) - (q2*c) / d +
                            (q2**2 * c * dy**2) / (d**2))
      c_xx_temp[0, 1, :] = ((q2**2 * c * dx * dy) / (d**2) + (q2*c*dx*dy) /
                            (d**3))
      c_xx_temp[1, 0, :] = ((q2**2 * c * dx * dy) / (d**2) + (q2*c*dx*dy) /
                            (d**3))

      c_x_temp[:, idx_ignore] = 0
      c_xx_temp[:, :, idx_ignore] = 0

      c_x += c_x_temp.sum(axis=-1)
      c_xx += c_xx_temp.sum(axis=-1)

    return c_x, c_xx
