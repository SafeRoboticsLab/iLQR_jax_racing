"""
Costs, gradients, and Hessians.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ECE346@Princeton (Zixu Zhang, Kai-Chieh Hsu, Duy P. Nguyen)
"""

import numpy as np
from typing import Tuple, List

from .constraints import Constraints

from functools import partial
from jax import jit, jacfwd, hessian
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp
import jax


class Cost:

  def __init__(self, config, safety=True):
    self.config = config
    self.soft_constraints = Constraints(config)
    self.safety = safety

    # Planning parameters.
    self.T = config.T  # planning horizon.
    self.N = config.N  # number of planning steps.
    self.dt = self.T / (self.N - 1)  # time step for each planning step.
    self.v_ref = config.V_REF  # reference velocity.

    # System parameters.
    self.dim_x = config.DIM_X
    self.dim_u = config.DIM_U

    # Track parameters.
    self.track_width_R = config.TRACK_WIDTH_R
    self.track_width_L = config.TRACK_WIDTH_L

    # Racing cost parameters.
    self.w_vel = config.W_VEL
    self.w_contour = config.W_CONTOUR
    self.w_theta = config.W_THETA
    self.w_accel = config.W_ACCEL
    self.w_delta = config.W_DELTA
    self.wheelbase = config.WHEELBASE
    self.track_offset = config.TRACK_OFFSET
    self.W_state = np.array([[self.w_contour, 0], [0, self.w_vel]])
    self.W_control = np.array([[self.w_accel, 0], [0, self.w_delta]])

    # Soft constraint parameters.
    self.q1_v = config.Q1_V
    self.q2_v = config.Q2_V
    self.q1_road = config.Q1_ROAD
    self.q2_road = config.Q2_ROAD
    self.q1_lat = config.Q1_LAT
    self.q2_lat = config.Q2_LAT
    self.q1_obs = config.Q1_OBS
    self.q2_obs = config.Q2_OBS
    self.barrier_thr = config.BARRIER_THR
    self.v_min = config.V_MIN
    self.v_max = config.V_MAX
    self.a_min = config.A_MIN
    self.a_max = config.A_MAX
    self.delta_min = config.DELTA_MIN
    self.delta_max = config.DELTA_MAX
    self.alat_min = config.ALAT_MIN
    self.alat_max = config.ALAT_MAX
    self.r = config.WIDTH / 2.
    self.road_thr = 0.025

    # Useful constant vectors.
    self.zeros = np.zeros((self.N))
    self.ones = np.ones((self.N))

    # Computes cost gradients using Jax.
    self.cx_x = jit(jacfwd(self.state_cost_stage_jitted, argnums=0))
    self.cp_x = self.progress_deriv_jitted
    self.cu_u = jit(jacfwd(self.control_cost_stage_jitted, argnums=0))
    self.rb_x = jit(jacfwd(self.road_boundary_cost_stage_jitted, argnums=0))
    self.la_x = jit(jacfwd(self.lat_acc_cost_stage_jitted, argnums=0))
    self.la_u_tmp = jit(jacfwd(self.lat_acc_cost_stage_jitted, argnums=1))
    self.vb_x = jit(jacfwd(self.vel_bound_cost_stage_jitted, argnums=0))

    # Vectorizes gradients using Jax.
    self.cx_x = jit(jax.vmap(self.cx_x, in_axes=(1, 1, 1), out_axes=(1)))
    self.cp_x = jit(jax.vmap(self.cp_x, in_axes=(1), out_axes=(1)))
    self.cu_u = jit(jax.vmap(self.cu_u, in_axes=(1), out_axes=(1)))
    self.rb_x = jit(jax.vmap(self.rb_x, in_axes=(1, 1, 1), out_axes=(1)))
    self.la_x = jit(jax.vmap(self.la_x, in_axes=(1, 1), out_axes=(1)))
    self.la_u = jit(jax.vmap(self.la_u_tmp, in_axes=(1, 1), out_axes=(1)))
    self.vb_x = jit(jax.vmap(self.vb_x, in_axes=(1), out_axes=(1)))

    # Computes cost Hessians and second-order derivatives using Jax.
    self.cx_xx = jit(hessian(self.state_cost_stage_jitted, argnums=0))
    self.cu_uu = jit(hessian(self.control_cost_stage_jitted, argnums=0))
    self.c_ux = jnp.zeros((self.dim_u, self.dim_x))
    self.rb_xx = jit(hessian(self.road_boundary_cost_stage_jitted, argnums=0))
    self.la_xx = jit(hessian(self.lat_acc_cost_stage_jitted, argnums=0))
    self.la_uu = jit(hessian(self.lat_acc_cost_stage_jitted, argnums=1))
    self.la_ux = jit(jacfwd(self.la_u_tmp, argnums=0))
    self.vb_xx = jit(hessian(self.vel_bound_cost_stage_jitted, argnums=0))

    # Vectorizes Hessians and second-order derivatives using Jax.
    self.cx_xx = jit(jax.vmap(self.cx_xx, in_axes=(1, 1, 1), out_axes=(2)))
    self.cu_uu = jit(jax.vmap(self.cu_uu, in_axes=(1), out_axes=(2)))
    self.rb_xx = jit(jax.vmap(self.rb_xx, in_axes=(1, 1, 1), out_axes=(2)))
    self.la_xx = jit(jax.vmap(self.la_xx, in_axes=(1, 1), out_axes=(2)))
    self.la_uu = jit(jax.vmap(self.la_uu, in_axes=(1, 1), out_axes=(2)))
    self.la_ux = jit(jax.vmap(self.la_ux, in_axes=(1, 1), out_axes=(2)))
    self.vb_xx = jit(jax.vmap(self.vb_xx, in_axes=(1), out_axes=(2)))

    # # Vectorizes soft constraint cost gradient and Hessians.
    # self.lat_acc_deriv_vmap = jit(
    #     jax.vmap(
    #         self.lat_acc_deriv_jitted, in_axes=(1, 1),
    #         out_axes=(1, 2, 1, 2, 2)
    #     )
    # )
    # self.road_boundary_deriv_vmap = jit(
    #     jax.vmap(
    #         self.road_boundary_deriv_jitted, in_axes=(1, 1, 1),
    #         out_axes=(1, 2)
    #     )
    # )
    # self.vel_bound_deriv_vmap = jit(
    #     jax.vmap(self.vel_bound_deriv_jitted, in_axes=(1), out_axes=(1, 2))
    # )

  def update_obs(self, frs_list: List):
    """
    Updates the obstacle list.

    Args:
        frs_list (List): obstacle list.
    """
    self.soft_constraints.update_obs(frs_list)

  def get_cost(
      self, states: np.ndarray, controls: np.ndarray, closest_pts: np.ndarray, slope: np.ndarray,
      theta: np.ndarray
  ) -> np.ndarray:
    """
    Calculates the cost given planned states and controls.

    Args:
        states (np.ndarray): 4xN array of planned trajectory.
        controls (np.ndarray): 2xN array of planned control.
        closest_pts (np.ndarray): 2xN array of each state's closest point [x,y]
            on the track.
        slope (np.ndarray): 1xN array of track's slopes (rad) at closest
            points.
        theta (np.ndarray): 1xN array of the progress at each state.

    Returns:
        np.ndarray: total cost.
    """
    transform = np.array([[np.sin(slope), -np.cos(slope), self.zeros, self.zeros],
                          [self.zeros, self.zeros, self.ones, self.zeros]])

    ref_states = np.zeros_like(states)
    ref_states[0, :] = closest_pts[0, :] + np.sin(slope) * self.track_offset
    ref_states[1, :] = closest_pts[1, :] - np.cos(slope) * self.track_offset
    ref_states[2, :] = self.v_ref

    error = states - ref_states
    Q_trans = np.einsum(
        'abn, bcn->acn', np.einsum('dan, ab -> dbn', transform.transpose(1, 0, 2), self.W_state),
        transform
    )

    c_state = np.einsum('an, an->n', error, np.einsum('abn, bn->an', Q_trans, error))
    c_progress = -self.w_theta * np.sum(theta)

    c_control = np.einsum('an, an->n', controls, np.einsum('ab, bn->an', self.W_control, controls))

    c_control[-1] = 0

    c_constraint = self.soft_constraints.get_cost(states, controls, closest_pts, slope)

    if not self.safety:
      c_constraint = 0

    J = np.sum(c_state + c_constraint + c_control) + c_progress

    return J

  def get_derivatives_jax(
      self, states: ArrayImpl, controls: ArrayImpl, closest_pts: ArrayImpl, slopes: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Calculates Jacobian and Hessian of the overall cost using Jax.

    Args:
        states (ArrayImpl): current states of the shape (dim_x, N).
        controls (ArrayImpl): current controls of the shape (dim_u, N).
        closest_pts (ArrayImpl): each state's closest point [x,y]
            on the track (2, N).
        slopes (ArrayImpl): track's slopes (rad) at closest points (1, N).

    Returns:
        ArrayImpl: c_x of the shape (dim_x, N).
        ArrayImpl: c_xx of the shape (dim_x, dim_x, N).
        ArrayImpl: c_u of the shape (dim_u, N).
        ArrayImpl: c_uu of the shape (dim_u, dim_u, N).
        ArrayImpl: c_ux of the shape (dim_u, dim_x, N).
    """
    # Obtains racing cost gradients and Hessians
    c_x_cost = self.cx_x(states, closest_pts, slopes)
    c_xx_cost = self.cx_xx(states, closest_pts, slopes)

    c_p_cost = self.cp_x(slopes)

    c_u_cost = self.cu_u(controls)
    c_uu_cost = self.cu_uu(controls)

    q = c_x_cost + c_p_cost
    Q = c_xx_cost
    r = c_u_cost
    R = c_uu_cost
    S = np.zeros((self.dim_u, self.dim_x, states.shape[1]))

    # Obtains soft constraint cost gradients and Hessians.
    # Analytical.
    if self.safety:
      c_x_obs, c_xx_obs = self.soft_constraints.get_obs_derivatives(np.asarray(states))

    c_x_rb = self.rb_x(states, closest_pts, slopes)
    c_xx_rb = self.rb_xx(states, closest_pts, slopes)

    # Jax-autodiff'd.
    c_x_lat = self.la_x(states, controls)
    c_xx_lat = self.la_xx(states, controls)
    c_u_lat = self.la_u(states, controls)
    c_uu_lat = self.la_uu(states, controls)
    c_ux_lat = self.la_ux(states, controls)

    c_x_v = self.vb_x(states)
    c_xx_v = self.vb_xx(states)

    if self.safety:
      q += c_x_lat + c_x_v + c_x_rb + c_x_obs
      Q += c_xx_lat + c_xx_v + c_xx_rb + c_xx_obs
    else:
      q += c_x_lat + c_x_v + c_x_rb
      Q += c_xx_lat + c_xx_v + c_xx_rb
    r += c_u_lat
    R += c_uu_lat
    S += c_ux_lat

    return q, Q, r, R, S

  # --------------------------- Jitted racing costs ----------------------------
  @partial(jit, static_argnums=(0,))
  def state_cost_stage_jitted(
      self, state: ArrayImpl, closest_pt: ArrayImpl, slope: ArrayImpl
  ) -> ArrayImpl:
    """
    Computes the stage state cost.

    Args:
        state (ArrayImpl): (dim_x,)
        closest_pt (ArrayImpl): (2,)
        slope (ArrayImpl): scalar

    Returns:
        ArrayImpl: cost (scalar)
    """
    sr = jnp.sin(slope)[0]
    cr = jnp.cos(slope)[0]
    transform = jnp.array([[sr, -cr, 0., 0.], [0., 0., 1., 0.]])
    Q = transform.T @ self.W_state @ transform
    ref_state = jnp.zeros((self.dim_x,))
    ref_state = ref_state.at[0].set(closest_pt[0] + sr * self.track_offset)
    ref_state = ref_state.at[1].set(closest_pt[1] - cr * self.track_offset)
    ref_state = ref_state.at[2].set(self.v_ref)
    return (state - ref_state).T @ Q @ (state-ref_state)

  @partial(jit, static_argnums=(0,))
  def control_cost_stage_jitted(self, control: ArrayImpl) -> ArrayImpl:
    """
    Computes the stage control cost c(u) = u.T @ R @ u, where u is the control.

    Args:
        control (ArrayImpl): (dim_u,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    return control.T @ self.W_control @ control

  @partial(jit, static_argnums=(0,))
  def progress_deriv_jitted(self, slope: ArrayImpl) -> ArrayImpl:
    """
    Computes the derivative of the progress cost w.r.t. the slope.

    Args:
        slope (ArrayImpl): scalar

    Returns:
        ArrayImpl: (dim_x,)
    """
    sr = jnp.sin(slope)[0]
    cr = jnp.cos(slope)[0]
    return -self.w_theta * jnp.hstack((cr, sr, 0, 0))

  # ----------------------- Jitted soft constraint costs -----------------------
  @partial(jit, static_argnums=(0,))
  def lat_acc_cost_stage_jitted(self, state: ArrayImpl, control: ArrayImpl) -> ArrayImpl:
    '''
    Calculates the lateral acceleration soft constraint cost.

    Args:
        state (ArrayImpl): (dim_x,)
        control (ArrayImpl): (dim_u,)

    Returns:
        ArrayImpl: dtype=float32
    '''
    accel = state[2]**2 * jnp.tan(control[1]) / self.wheelbase
    error_ub = accel - self.alat_max
    error_lb = self.alat_min - accel

    c_lat_ub = self.q1_lat * (jnp.exp(self.q2_lat * error_ub) - 1.)
    c_lat_lb = self.q1_lat * (jnp.exp(self.q2_lat * error_lb) - 1.)
    return c_lat_lb + c_lat_ub

  @partial(jit, static_argnums=(0,))
  def road_boundary_cost_stage_jitted(
      self, state: ArrayImpl, closest_pt: ArrayImpl, slope: ArrayImpl
  ) -> ArrayImpl:
    """
    Calculates the road boundary soft constraint cost.

    Args:
        state (ArrayImpl): (dim_x,)
        closest_pt (ArrayImpl): (2,)
        slope (ArrayImpl): (1,)

    Returns:
        ArrayImpl: dtype=float32
    """
    sr = jnp.sin(slope)
    cr = jnp.cos(slope)
    dx = state[0] - closest_pt[0]
    dy = state[1] - closest_pt[1]
    dis = (sr*dx - cr*dy)[0]

    _min_val = -self.road_thr * self.q2_road
    _max_val = self.barrier_thr

    # Right bound.
    b_r = dis - (self.track_width_R - self.r)
    c_r = (self.q1_road * jnp.exp(jnp.clip(self.q2_road * b_r, _min_val, _max_val)))

    # Left bound.
    b_l = -dis - (self.track_width_L - self.r)
    c_l = (self.q1_road * jnp.exp(jnp.clip(self.q2_road * b_l, _min_val, _max_val)))
    return c_l + c_r

  @partial(jit, static_argnums=(0,))
  def vel_bound_cost_stage_jitted(self, state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the velocity bound soft constraint cost.

    Args:
        state (ArrayImpl): (dim_x,)

    Returns:
        ArrayImpl: dtype=float32
    """
    cons_v_min = self.v_min - state[2]
    cons_v_max = state[2] - self.v_max
    barrier_v_min = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_min, None, self.barrier_thr))
    barrier_v_max = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_max, None, self.barrier_thr))
    return barrier_v_min + barrier_v_max

  # ----------- Jitted analytical derivatives (for comparison only) ------------
  @partial(jit, static_argnums=(0,))
  def lat_acc_deriv_jitted(
      self, state: ArrayImpl, control: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Calculates the Jacobian and Hessian of lateral acc. soft constraint cost.

    Args:
        state (ArrayImpl): of the shape (dim_x,).
        control (ArrayImpl): of the shape (dim_u,).

    Returns:
        ArrayImpl: c_x of the shape (dim_x,).
        ArrayImpl: c_xx of the shape (dim_x, dim_x).
        ArrayImpl: c_u of the shape (dim_u,).
        ArrayImpl: c_uu of the shape (dim_u, dim_u).
        ArrayImpl: c_ux of the shape (dim_u, dim_x).
    """
    dim_x = self.dim_x
    wheelbase = self.wheelbase
    q1_lat = self.q1_lat
    q2_lat = self.q2_lat
    barrier_thr = self.barrier_thr

    c_x = jnp.zeros((dim_x,))
    c_xx = jnp.zeros((dim_x, dim_x))
    c_u = jnp.zeros((2,))
    c_uu = jnp.zeros((2, 2))
    c_ux = jnp.zeros((2, dim_x))

    accel = state[2]**2 * jnp.tan(control[1]) / wheelbase
    error_ub = accel - self.alat_max
    error_lb = self.alat_min - accel

    cost_a_lat_min = q1_lat * jnp.exp(jnp.clip(q2_lat * error_lb, None, barrier_thr))
    cost_a_lat_max = q1_lat * jnp.exp(jnp.clip(q2_lat * error_ub, None, barrier_thr))

    da_dx = 2 * state[2] * jnp.tan(control[1]) / wheelbase
    da_dxx = 2 * jnp.tan(control[1]) / wheelbase

    da_du = state[2]**2 / (jnp.cos(control[1])**2 * wheelbase)
    da_duu = (state[2]**2 * jnp.sin(control[1]) / (jnp.cos(control[1])**3 * wheelbase))

    da_dux = 2 * state[2] / (jnp.cos(control[1])**2 * wheelbase)

    c_x = c_x.at[2].set(q2_lat * (cost_a_lat_max-cost_a_lat_min) * da_dx)
    c_u = c_u.at[1].set(q2_lat * (cost_a_lat_max-cost_a_lat_min) * da_du)

    c_xx = c_xx.at[2, 2].set(
        q2_lat**2 * (cost_a_lat_max+cost_a_lat_min) * da_dx**2
        + q2_lat * (cost_a_lat_max-cost_a_lat_min) * da_dxx
    )
    c_uu = c_uu.at[1, 1].set(
        q2_lat**2 * (cost_a_lat_max+cost_a_lat_min) * da_du**2
        + q2_lat * (cost_a_lat_max-cost_a_lat_min) * da_duu
    )

    c_ux = c_ux.at[1, 2].set(
        q2_lat**2 * (cost_a_lat_max+cost_a_lat_min) * da_dx * da_du
        + q2_lat * (cost_a_lat_max-cost_a_lat_min) * da_dux
    )
    return c_x, c_xx, c_u, c_uu, c_ux

  @partial(jit, static_argnums=(0,))
  def road_boundary_deriv_jitted(self, state: ArrayImpl, closest_pt: ArrayImpl,
                                 slope: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Calculates the Jacobian and Hessian of road boundary soft constraint cost.

    Args:
        state (ArrayImpl): (dim_x,)
        closest_pt (ArrayImpl): (2,)
        slope (ArrayImpl): (1,)

    Returns:
        ArrayImpl: (dim_x,)
        ArrayImpl: (dim_x, dim_x)
    """
    dim_x = self.dim_x
    sr = jnp.sin(slope)
    cr = jnp.cos(slope)
    dx = state[0] - closest_pt[0]
    dy = state[1] - closest_pt[1]
    dis = sr*dx - cr*dy
    transform = jnp.array([sr, -cr])

    cons_road_r = dis - (self.track_width_R - self.r)
    c_x_r = jnp.zeros((dim_x,))
    c_xx_r = jnp.zeros((dim_x, dim_x))
    _c_x_r, _c_xx_r = barrier_function_jitted(
        q1=self.q1_road, q2=self.q2_road, cons=cons_road_r, cons_dot=transform, cons_min=None,
        cons_max=self.barrier_thr
    )
    c_x_r = c_x_r.at[:2].set(jnp.ravel(_c_x_r))
    c_xx_r = c_xx_r.at[:2, :2].set(_c_xx_r)

    cons_road_l = -dis - (self.track_width_L - self.r)
    c_x_l = jnp.zeros((dim_x,))
    c_xx_l = jnp.zeros((dim_x, dim_x))
    _c_x_l, _c_xx_l = barrier_function_jitted(
        q1=self.q1_road, q2=self.q2_road, cons=cons_road_l, cons_dot=-transform, cons_min=None,
        cons_max=self.barrier_thr
    )
    c_x_l = c_x_l.at[:2].set(jnp.ravel(_c_x_l))
    c_xx_l = c_xx_l.at[:2, :2].set(_c_xx_l)

    return c_x_r + c_x_l, c_xx_r + c_xx_l

  @partial(jit, static_argnums=(0,))
  def vel_bound_deriv_jitted(self, state: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Calculates the Jacobian and Hessian of velocity soft constraint cost.

    Args:
        state (ArrayImpl): (dim_x,)

    Returns:
        ArrayImpl: (dim_x,)
        ArrayImpl: (dim_x, dim_x)
    """
    dim_x = self.dim_x
    transform = 1.
    c_x = jnp.zeros((dim_x,))
    c_xx = jnp.zeros((dim_x, dim_x))

    cons_v_min = self.v_min - state[2]
    cons_v_max = state[2] - self.v_max

    _c_x_min, _c_xx_min = barrier_function_jitted(
        self.q1_v, self.q2_v, cons_v_min, -transform, cons_max=self.barrier_thr
    )
    _c_x_max, _c_xx_max = barrier_function_jitted(
        self.q1_v, self.q2_v, cons_v_max, transform, cons_max=self.barrier_thr
    )
    c_x = c_x.at[2].set(_c_x_min + _c_x_max)
    c_xx = c_xx.at[2, 2].set((_c_xx_min + _c_xx_max)[0, 0])

    return c_x, c_xx


@jit
def barrier_function_jitted(
    q1: ArrayImpl, q2: ArrayImpl, cons: ArrayImpl, cons_dot: ArrayImpl, cons_min: ArrayImpl = None,
    cons_max: ArrayImpl = None
) -> Tuple[ArrayImpl, ArrayImpl]:
  """
  Computes the barrier function with clipping.

  Args:
      q1 (ArrayImpl): scalar
      q2 (ArrayImpl): scalar
      cons (ArrayImpl): scalar
      cons_dot (ArrayImpl): N-by-1
      cons_min (ArrayImpl): scalar
      cons_max (ArrayImpl): scalar

  Returns:
      ArrayImpl: N-by-1
      ArrayImpl: N-by-N
  """
  tmp = jnp.clip(q2 * cons, cons_min, cons_max)
  b = q1 * (jnp.exp(tmp))

  b_dot = q2 * b * cons_dot
  b_ddot = (q2**2) * b * jnp.outer(cons_dot, cons_dot)

  return b_dot, b_ddot
