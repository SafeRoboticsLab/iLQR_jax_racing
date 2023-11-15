"""
Dynamics.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ECE346@Princeton (Zixu Zhang, Kai-Chieh Hsu, Duy P. Nguyen)
TODOs:
  - Make state & ctrl indices global, static variables, e.g. self._px_idx = 0
"""

from typing import Optional, Tuple
import numpy as np

from functools import partial
from jax import jit, jacfwd
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp
import jax


class Dynamics:

  def __init__(self, config):
    self.dim_x = 4
    self.dim_u = 2

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

    # Useful constants.
    self.zeros = np.zeros((self.N))
    self.ones = np.ones((self.N))

    # Computes Jacobian matrices using Jax.
    self.jac_f = jit(jacfwd(self.dct_time_dyn, argnums=[0, 1]))

    # Vectorizes Jacobians using Jax.
    self.jac_f = jit(jax.vmap(self.jac_f, in_axes=(1, 1), out_axes=(2, 2)))

  def forward_step(
      self, state: np.ndarray, control: np.ndarray, step: Optional[int] = 1,
      noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif'
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the next state of the vehicle given the current state and
    control input state.

    Args:
        state (np.ndarray): (4, ) array [X, Y, V, psi].
        control (np.ndarray): (2, ) array [a, delta].
        step (int, optional): The number of segements to forward the
            dynamics. Defaults to 1.
        noise (np.ndarray, optional): The ball radius or standard
            deviation of the Gaussian noise. The magnitude should be in the
            sense of self.dt. Defaults to None.
        noise_type(str, optional): Uniform or Gaussian. Defaults to 'unif'.

    Returns:
        np.ndarray: next state.
        np.ndarray: clipped control.
    """
    # Clips the controller values between min and max accel and steer values.
    accel = np.clip(control[0], self.a_min, self.a_max)
    delta = np.clip(control[1], self.delta_min, self.delta_max)
    control_clip = np.array([accel, delta])
    next_state = state
    dt_step = self.dt / step

    for _ in range(step):
      # State: [x, y, v, psi]
      d_x = ((next_state[2] * dt_step + 0.5 * accel * dt_step**2) * np.cos(next_state[3]))
      d_y = ((next_state[2] * dt_step + 0.5 * accel * dt_step**2) * np.sin(next_state[3]))
      d_v = accel * dt_step
      d_psi = ((next_state[2] * dt_step + 0.5 * accel * dt_step**2) * np.tan(delta)
               / self.wheelbase)
      next_state = next_state + np.array([d_x, d_y, d_v, d_psi])
      if noise is not None:
        T = np.array([[np.cos(next_state[-1]), np.sin(next_state[-1]), 0, 0],
                      [-np.sin(next_state[-1]),
                       np.cos(next_state[-1]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        if noise_type == 'unif':
          rv = np.random.rand(4) - 0.5
        else:
          rv = np.random.normal(size=(4))
        next_state = next_state + (T@noise) * rv / step

      # Clip the velocity.
      next_state[2] = np.clip(next_state[2], 0, None)

    return next_state, control_clip

  # ---------------------------- Jitted functions ------------------------------
  @partial(jit, static_argnums=(0,))
  def dct_time_dyn(self, state: ArrayImpl, control: ArrayImpl) -> ArrayImpl:
    """
    Computes the one-step time evolution of the system.

    Args:
        state (ArrayImpl): (4,) jnp array [x, y, v, psi].
        control (ArrayImpl): (2,) jnp array [a, delta].

    Returns:
        ArrayImpl: next state.
    """
    d_x = ((state[2] * self.dt + 0.5 * control[0] * self.dt**2) * jnp.cos(state[3]))
    d_y = ((state[2] * self.dt + 0.5 * control[0] * self.dt**2) * jnp.sin(state[3]))
    d_v = control[0] * self.dt
    d_psi = ((state[2] * self.dt + 0.5 * control[0] * self.dt**2) * jnp.tan(control[1])
             / self.wheelbase)
    return state + jnp.hstack((d_x, d_y, d_v, d_psi))

  @partial(jit, static_argnums=(0,))
  def integrate_forward_jax(self, state: ArrayImpl,
                            control: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Computes the next state.

    Args:
        state (ArrayImpl): (4,) jnp array [x, y, v, psi].
        control (ArrayImpl): (2,) jnp array [a, delta].

    Returns:
        state_next: ArrayImpl
        control_next: ArrayImpl
    """
    # Clips the control values with their limits.
    accel = jnp.clip(control[0], self.a_min, self.a_max)
    delta = jnp.clip(control[1], self.delta_min, self.delta_max)

    # Integrates the system one-step forward in time using the Euler method.
    control_clip = jnp.hstack((accel, delta))
    state_next = self.dct_time_dyn(state, control_clip)

    return state_next, control_clip

  @partial(jit, static_argnums=(0,))
  def get_AB_matrix_jax(self, nominal_states: ArrayImpl,
                        nominal_controls: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
      Returns the linearized 'A' and 'B' matrix of the ego vehicle around
      nominal states and controls.

      Args:
          nominal_states (ArrayImpl): (nx, N) states along the nominal traj.
          nominal_controls (ArrayImpl): (nu, N) controls along the traj.

      Returns:
          ArrayImpl: the Jacobian of next state w.r.t. the current state.
          ArrayImpl: the Jacobian of next state w.r.t. the current control.
      """
    A, B = self.jac_f(nominal_states, nominal_controls)
    return A, B
