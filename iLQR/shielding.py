"""
iLQR-based shielding.

Please contact the author(s) of this library if you have any questions.
Author:  Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np

from jaxlib.xla_extension import ArrayImpl


class Shielding(object):

  def __init__(self, obs_list, config, N_sh=None):
    """
    Base class for all shielding mechanisms.

    Args:
        obs_list (list): list of ellipsoidal obstacles.
        config (Struct): parameter data.
        N_sh (int): shielding rollout horizon.
    """

    self.obs_list = obs_list
    self.config = config
    self.coll_check_slack = config.COLL_CHECK_SLACK
    self.N_sh = N_sh
    self.sh_flag = False

  def is_collision(self, states: ArrayImpl) -> bool:
    """
    Checks collision with the static obstacles.

    Args:
        states (ArrayImpl): state trajectory of the ego agent.

    Returns:
        bool: True if there is a collision.
    """
    states = np.asarray(states)

    if len(states.shape) < 2:
      states = states[:, np.newaxis]

    if self.N_sh is None:
      N_sh = states.shape[1]
    else:
      N_sh = min(states.shape[1], self.N_sh)

    res = False
    for obs in self.obs_list:
      for k in range(N_sh):
        pos_k = states[:2, k]
        pos_k = pos_k[:, np.newaxis]
        if obs.is_internal(pos_k, slack=self.coll_check_slack):
          return True
    return res


class NaiveSwerving(Shielding):

  def __init__(self, config, obs_list, dynamics, max_deacc, N_sh=None):
    """
    Naive swerving shielding strategy: maximum steering and deacceleration.

    Args:
        obs_list (list): list of ellipsoidal obstacles.
        config (Struct): parameter data.
        dynamics (Dynamics): dynamical system.
        max_deacc (float): maximum deacceleration.
        N_sh (int): shielding rollout horizon.
    """

    self.states = None
    self.controls_init = np.zeros((2, config.N))
    self.max_deacc = max_deacc
    self.delta_min = config.DELTA_MIN
    self.delta_max = config.DELTA_MAX
    self.dynamics = dynamics
    super(NaiveSwerving, self).__init__(obs_list, config, N_sh)

  def run(self, x: ArrayImpl, u_nominal: ArrayImpl) -> np.ndarray:
    """
    Runs the naive swerving shielding.

    Args:
        x (ArrayImpl): current state.
        u_nominal (ArrayImpl): nominal control.

    Returns:
        np.ndarray: shielded control.
    """

    x = np.asarray(x)
    u_nominal = np.asarray(u_nominal)

    # Resets the shielding flag.
    self.sh_flag = False

    # One-step simulation with robot's nominal policy.
    x_next_nom = self.dynamics.forward_step(x, u_nominal)[0]

    success_delta_min = True

    if self.is_collision(x_next_nom):
      self.sh_flag = True
    else:
      # Computes iLQR shielding policies starting with x_next_nom.
      # 1. First try delta_min.
      u_sh = np.array([self.max_deacc, self.delta_min])
      x_sh = x_next_nom
      for _ in range(self.N_sh - 1):
        x_sh = self.dynamics.forward_step(x_sh, u_sh)[0]
        if self.is_collision(x_sh):
          success_delta_min = False
          break

      # 2. If delta_min fails, then try delta_max.
      if not success_delta_min:
        u_sh = np.array([self.max_deacc, self.delta_max])
        x_sh = x_next_nom
        for _ in range(self.N_sh - 1):
          x_sh = self.dynamics.forward_step(x_sh, u_sh)[0]
          if self.is_collision(x_sh):
            self.sh_flag = True
            break

    # Shielding is needed.
    if self.sh_flag:
      # print("Shielding override.")
      if success_delta_min:
        u_sh = np.array([self.max_deacc, self.delta_min])

      else:
        u_sh = np.array([self.max_deacc, self.delta_max])

      states_sh = x[:, np.newaxis]
      for _ in range(self.N_sh - 1):
        x = self.dynamics.forward_step(x, u_sh)[0]
        states_sh = np.hstack((states_sh, x[:, np.newaxis]))

      self.states = states_sh

      return u_sh

    # Shielding is not needed.
    else:
      return u_nominal


class ILQshielding(Shielding):

  def __init__(self, config, solver, obs_list, obs_list_timed, N_sh=None):
    """
    ILQR-based shielding policy.

    Args:
        config (Struct): parameter data.
        solver (iLQR): iLQR solver.
        obs_list (list): list of ellipsoidal obstacles.
        obs_list_timed (list): list of time-varying obstacles.
        N_sh (int): shielding rollout horizon.
    """

    self.solver = solver
    self.obs_list_timed = obs_list_timed
    self.states = None
    self.controls_init = np.zeros((2, config.N))
    super(ILQshielding, self).__init__(obs_list, config, N_sh)

  def run(self, x: ArrayImpl, u_nominal: ArrayImpl) -> np.ndarray:
    """
    Runs the iLQR-based shielding.

    Args:
        x (ArrayImpl): current state.
        u_nominal (ArrayImpl): nominal control.

    Returns:
        np.ndarray: shielded control.
    """

    # Resets the shielding flag.
    self.sh_flag = False

    # Computes iLQR shielding policies starting with x.
    states_sh, controls_sh, _, _, _, _, _, _ = (
        self.solver.solve(x, controls=self.controls_init, obs_list=self.obs_list_timed)
    )
    self.states = states_sh

    # One-step simulation with robot's nominal policy.
    x_next_nom = self.solver.dynamics.forward_step(x, u_nominal)[0]
    if self.is_collision(x_next_nom):
      self.sh_flag = True
    else:
      # Computes iLQR shielding policies starting with x_next_nom.
      states, _, _, _, _, _, _, _ = (
          self.solver.solve(x_next_nom, controls=self.controls_init, obs_list=self.obs_list_timed)
      )
      if self.is_collision(states):
        self.sh_flag = True

    # Updates the init. control signal for warmstart of next receding horizon.
    self.controls_init[:, :-1] = controls_sh[:, 1:]

    # Shielding is needed.
    if self.sh_flag:
      # print("Shielding override.")
      controls_sh = np.asarray(controls_sh[:, 0])
      if controls_sh[1] <= 0:
        u_sh = np.array([controls_sh[0], self.config.DELTA_MIN])
      else:
        u_sh = np.array([controls_sh[0], self.config.DELTA_MAX])
      return u_sh

    # Shielding is not needed.
    else:
      return np.asarray(u_nominal)
