"""
Constraints.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ECE346@Princeton (Zixu Zhang, Kai-Chieh Hsu, Duy P. Nguyen)
"""

import numpy as np
from typing import List, Tuple

from .ellipsoid_obj import EllipsoidObj


class Constraints:

  def __init__(self, config):
    self.T = config.T  # Planning Time Horizon
    self.N = config.N  # number of planning steps
    self.dt = self.T / (self.N - 1)  # time step for each planning step
    self.wheelbase = config.WHEELBASE  # vehicle chassis length
    self.delta_min = config.DELTA_MIN  # min steering angle rad
    self.delta_max = config.DELTA_MAX  # max steering angle rad
    self.a_min = config.A_MIN  # min longitudial accel
    self.a_max = config.A_MAX  # max longitudial accel
    self.v_min = config.V_MIN  # min velocity
    self.v_max = config.V_MAX  # max velocity
    self.alat_min = config.ALAT_MIN  # min lateral accel
    self.alat_max = config.ALAT_MAX  # max lateral accel
    self.track_width_L = config.TRACK_WIDTH_L
    self.track_width_R = config.TRACK_WIDTH_R

    # System parameters.
    self.dim_x = config.DIM_X
    self.dim_u = config.DIM_U

    # Parameter for barrier functions.
    self.q1_v = config.Q1_V
    self.q2_v = config.Q2_V

    self.q1_road = config.Q1_ROAD
    self.q2_road = config.Q2_ROAD

    self.q1_lat = config.Q1_LAT
    self.q2_lat = config.Q2_LAT

    self.q1_obs = config.Q1_OBS
    self.q2_obs = config.Q2_OBS

    self.exp_ub = config.EXP_UB

    # Useful constants.
    self.zeros = np.zeros((self.N))
    self.ones = np.ones((self.N))

    self.gamma = 0.9
    ego_a = config.LENGTH / 2.0
    self.r = ego_b = config.WIDTH / 2.0
    wheelbase = config.WHEELBASE
    ego_q = np.array([wheelbase / 2, 0])[:, np.newaxis]
    ego_Q = np.diag([ego_a * ego_a, ego_b * ego_b])
    self.ego = EllipsoidObj(q=ego_q, Q=ego_Q)
    self.ego_ell = None
    self.obs_list = None

  def update_obs(self, frs_list: List):
    """
    Updates the obstacle list.

    Args:
        frs_list (List): obstacle list.
    """
    self.obs_list = frs_list

  def state2ell(self, states: np.ndarray) -> List[EllipsoidObj]:
    """
    Converts a state variable to an ellipsoid object.

    Args:
        states (np.ndarray): 4xN array of state.

    Returns:
        List[EllipsoidObj]: ellipsoid object.
    """
    ego_ell = []
    for i in range(self.N):
      theta = states[3, i]
      d = states[:2, i][:, np.newaxis]
      R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
      temp = self.ego @ R
      temp.add(d)
      ego_ell.append(temp)
    return ego_ell

  def get_cost(
      self, states: np.ndarray, controls: np.ndarray, closest_pt: np.ndarray, slope: np.ndarray
  ) -> np.ndarray:
    """
    Computes the total soft constraint cost.

    Args:
        states (np.ndarray): 4xN array of state.
        controls (np.ndarray): 2xN array of control input.
        closest_pts (np.ndarray): 2xN array of each state's closest point [x,y]
            on the track.
        slope (np.ndarray): 1xN array of track's slopes (rad) at closest
            points.

    Returns:
        np.ndarray: total soft constraint cost.
    """

    # Road Boundary constarint.
    c_boundary = self._road_boundary_cost(states, closest_pt, slope)

    # Obstacle constraints.
    c_obs = np.zeros(self.N)
    if len(self.obs_list) > 0:
      # Patch footprint around state trajectory.
      self.ego_ell = self.state2ell(states)
      for i in range(self.N):
        ego_i = self.ego_ell[i]
        for obs_j in self.obs_list:  # obs_j is a list of obstacles.
          obs_j_i = obs_j[i]  # Get the ith obstacle in list obs_j.
          c_obs[i] += self.gamma**i * ego_i.obstacle_cost(obs_j_i, self.q1_obs, self.q2_obs)

    # Minimum velocity constarint.
    c_vel = self.q1_v * (np.exp(-states[2, :] * self.q2_v))

    # Lateral Acceleration constraint.
    accel = states[2, :]**2 * np.tan(controls[1, :]) / self.wheelbase
    error_ub = accel - self.alat_max
    error_lb = self.alat_min - accel

    b_ub = self.q1_lat * (np.exp(np.clip(self.q2_lat * error_ub, None, self.exp_ub)) - 1)
    b_lb = self.q1_lat * (np.exp(np.clip(self.q2_lat * error_lb, None, self.exp_ub)) - 1)
    c_lat = b_lb + b_ub

    return c_vel + c_boundary + c_lat + c_obs

  def _road_boundary_cost(
      self, states: np.ndarray, closest_pt: np.ndarray, slope: np.ndarray
  ) -> np.ndarray:
    """
    Computes the road boundary cost.

    Args:
        states (np.ndarray): 4xN array of state.
        closest_pt (np.ndarray): 2xN array of each state's closest point [x,y]
            on the track.
        slope (np.ndarray): 1xN array of track's slopes (rad) at closest
            points.

    Returns:
        np.ndarray: road boundary cost.
    """
    dx = states[0, :] - closest_pt[0, :]
    dy = states[1, :] - closest_pt[1, :]

    sr = np.sin(slope)
    cr = np.cos(slope)
    dis = sr*dx - cr*dy
    # Right bound.
    b_r = dis - (self.track_width_R - self.r)

    c_r = self.q1_road * np.exp(np.clip(self.q2_road * b_r, -0.025 * self.q2_road, 20))
    # Left Bound.
    b_l = -dis - (self.track_width_L - self.r)

    c_l = self.q1_road * np.exp(np.clip(self.q2_road * b_l, -0.025 * self.q2_road, 20))

    return c_l + c_r

  def get_obs_derivatives(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Jacobian and Hessian of the obstacle avoidance soft
    constraint cost.

    Args:
        states (np.ndarray): 4xN array of state.

    Returns:
        np.ndarray: Jacobian vector.
        np.ndarray: Hessian matrix.
    """
    c_x_obs = np.zeros((self.dim_x, self.N))
    c_xx_obs = np.zeros((self.dim_x, self.dim_x, self.N))

    if len(self.obs_list) > 0:
      for i in range(self.N):
        ego_i = self.ego_ell[i]
        for obs_j in self.obs_list:
          obs_j_i = obs_j[i]
          c_x_obs_temp, c_xx_obs_temp = ego_i.obstacle_derivative(
              states[:, i], self.wheelbase / 2, obs_j_i, self.q1_obs, self.q2_obs
          )
          c_x_obs[:, i] += self.gamma**i * c_x_obs_temp
          c_xx_obs[:, :, i] += self.gamma**i * c_xx_obs_temp

    return c_x_obs, c_xx_obs
