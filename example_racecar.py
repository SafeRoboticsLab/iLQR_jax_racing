"""
Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ECE346@Princeton (Zixu Zhang, Kai-Chieh Hsu, Duy P. Nguyen)
Checklist for slow JAX computation:
  - Mixing DeviceArray with numpy array.
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from IPython.display import Image
import imageio
import argparse

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

from iLQR.utils import *
from iLQR.ilqr import iLQR
from iLQR.shielding import ILQshielding, NaiveSwerving
from iLQR.ellipsoid_obj import EllipsoidObj


def main(config_file):
  # Loads the config and track file.
  config = load_config(config_file)
  track = load_track(config)

  # Constructs static obstacles.
  static_obs_list = []
  obs_a = config.LENGTH / 2.0
  obs_b = config.WIDTH / 2.0

  obs_q1 = np.array([0, 5.6])[:, np.newaxis]
  obs_Q1 = np.diag([obs_a**2, obs_b**2])
  static_obs1 = EllipsoidObj(q=obs_q1, Q=obs_Q1)
  static_obs_list.append([static_obs1 for _ in range(config.N)])

  obs_q2 = np.array([-2.1, 4.0])[:, np.newaxis]
  obs_Q2 = np.diag([obs_b**2, obs_a**2])
  static_obs2 = EllipsoidObj(q=obs_q2, Q=obs_Q2)
  static_obs_list.append([static_obs2 for _ in range(config.N)])

  obs_q3 = np.array([-2.4, 2.0])[:, np.newaxis]
  obs_Q3 = np.diag([obs_b**2, obs_a**2])
  static_obs3 = EllipsoidObj(q=obs_q3, Q=obs_Q3)
  static_obs_list.append([static_obs3 for _ in range(config.N)])

  # Initialization.
  solver = iLQR(track, config, safety=False)  # Nominal racing iLQR
  solver_sh = iLQR(track, config, safety=True)  # Shielding safety backup iLQR

  # max_deacc = config.A_MIN / 3  # config.A_MIN, config.A_MIN / 2, config.A_MIN / 3
  # shielding = NaiveSwerving(
  #     config, [static_obs1, static_obs2, static_obs3], solver_sh.dynamics,
  #     max_deacc=max_deacc, N_sh=30
  # )

  shielding = ILQshielding(
      config, solver_sh, [static_obs1, static_obs2, static_obs3],
      static_obs_list, N_sh=15
  )

  pos0, psi0 = track.interp([2])  # Position and yaw on the track.
  x_cur = np.array([3.3, 4.0, 0, np.pi / 2])  # Initial state.
  # x_cur = np.array([pos0[0], pos0[1], 0, psi0[-1]])
  init_control = np.zeros((2, config.N))
  t_total = 0.
  plot_cover = False

  itr_receding = config.MAX_ITER_RECEDING  # The number of receding iterations.
  state_hist = np.zeros((4, itr_receding))
  control_hist = np.zeros((2, itr_receding))
  plan_hist = np.zeros((6, config.N, itr_receding))
  K_hist = np.zeros((2, 4, config.N - 1, itr_receding))
  fx_hist = np.zeros((4, 4, config.N, itr_receding))
  fu_hist = np.zeros((4, 2, config.N, itr_receding))

  # Specifies the folder to save figures.
  fig_prog_folder = os.path.join(config.OUT_FOLDER, "progress")
  os.makedirs(fig_prog_folder, exist_ok=True)

  # Define disturbances.
  sigma = np.array([
      config.SIGMA_X, config.SIGMA_Y, config.SIGMA_V, config.SIGMA_THETA
  ])

  # iLQR Planning.
  for i in range(itr_receding):
    # Plans the trajectory using iLQR.
    states, controls, t_process, status, _, K_closed_loop, fx, fu = (
        solver.solve(x_cur, controls=init_control, obs_list=static_obs_list)
    )

    # Shielding.
    control_sh = shielding.run(x=x_cur, u_nominal=controls[:, 0])

    # Executes the control.
    x_cur = solver.dynamics.forward_step(
        x_cur, control_sh, step=1, noise=sigma
    )[0]
    print(
        "[{}]: solver returns status {} and uses {:.3f}.".format(
            i, status, t_process
        ), end='\r'
    )
    if i > 0:  # Excludes JAX compilation time at the first time step.
      t_total += t_process

    # Records planning history, states and controls.
    plan_hist[:4, :, i] = states
    plan_hist[4:, :, i] = controls
    state_hist[:, i] = states[:, 0]
    control_hist[:, i] = controls[:, 0]

    K_hist[:, :, :, i] = K_closed_loop
    fx_hist[:, :, :, i] = fx
    fu_hist[:, :, :, i] = fu

    # Updates the nominal control signal for warmstart of next receding horizon.
    init_control[:, :-1] = controls[:, 1:]

    # Plots the current progress.
    plt.clf()
    track.plot_track()
    for static_obs in static_obs_list:
      plot_ellipsoids(
          plt.gca(), static_obs[0:1], arg_list=[dict(c='k', linewidth=1.)],
          dims=[0, 1], N=50, plot_center=False, use_alpha=False
      )
      if plot_cover:  # plot circles that cover the footprint.
        static_obs[0].plot_circ(plt.gca())
        solver.cost.soft_constraints.ego_ell[0].plot_circ(plt.gca())
    if shielding.sh_flag:
      plot_ellipsoids(
          plt.gca(), [solver.cost.soft_constraints.ego_ell[0]],
          arg_list=[dict(c='r')], dims=[0, 1], N=50, plot_center=False
      )
    else:
      plot_ellipsoids(
          plt.gca(), [solver.cost.soft_constraints.ego_ell[0]],
          arg_list=[dict(c='b')], dims=[0, 1], N=50, plot_center=False
      )
    plt.plot(states[0, :], states[1, :], linewidth=2, c='b')
    if shielding.sh_flag:
      plt.plot(
          shielding.states[0, :], shielding.states[1, :], linewidth=2, c='r'
      )
    sc = plt.scatter(
        state_hist[0, :i + 1], state_hist[1, :i + 1], s=24,
        c=state_hist[2, :i + 1], cmap=cm.jet, vmin=0, vmax=config.V_MAX,
        edgecolor='none', marker='o'
    )
    cbar = plt.colorbar(sc)
    cbar.set_label(r"velocity [$m/s$]", size=20)
    plt.axis('equal')
    plt.savefig(os.path.join(fig_prog_folder, str(i) + ".png"), dpi=200)

  plt.close('All')
  print("Planning uses {:.3f}.".format(t_total))

  # Makes an animation.
  gif_path = os.path.join(config.OUT_FOLDER, 'rollout.gif')
  with imageio.get_writer(gif_path, mode='I') as writer:
    for i in range(itr_receding):
      filename = os.path.join(fig_prog_folder, str(i) + ".png")
      image = imageio.imread(filename)
      writer.append_data(image)
  Image(open(gif_path, 'rb').read(), width=400)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("", "example_racecar.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
