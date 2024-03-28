# Safe Autonomous Racing using iLQR and Rollout-based Shielding: A JAX Implementation

[![License][license-shield]][license-url]
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/)
[![Colab Notebook][homepage-shield]][homepage-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/SafeRoboticsLab/iLQR_jax_racing_dev">
    <img src="experiments/ilqr_jax/rollout.gif" alt="Logo" width="600">
  </a>
  <!-- <h3 align="center">ILQR JAX Racing</h3> -->
  <p align="center">
    <!-- Safe Autonomous Racing using iLQR and Rollout-based Shielding: A JAX Implementation -->
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#dependencies">Dependencies</a></li>
    <li><a href="#example">Example</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#paper">Paper</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This repository implements a safe autonomous racing example using iLQR and rollout-based shielding, which relies on [JAX](https://github.com/google/jax) for real-time computation performance based on automatic differentiation and just-in-time (JIT) compilation.
The repo is primarily developed and maintained by [Haimin Hu](https://haiminhu.org/), a PhD student in the [Safe Robotics Lab](https://saferobotics.princeton.edu).
[Zixu Zhang](https://zzx9636.github.io/), [Kai-Chieh Hsu](https://kaichiehhsu.github.io/) and [Duy Nguyen](https://ece.princeton.edu/people/duy-phuong-nguyen) also contributed much to the repo (their original repo is [here](https://github.com/SafeRoboticsLab/PrincetonRaceCar_planning)).


## Dependencies

This repo depends on the following packages:
1. jax=0.4.19
2. jaxlib=0.4.16
3. matplotlib=3.5.1
4. numpy=1.21.5
5. pyspline=1.5.1
6. python=3.10
7. yaml=0.2.5


## Example
Please refer to the [Colab Notebook](https://colab.research.google.com/drive/1_3HgZx7LTBw69xH61Us70xI8HISUeFA7?usp=sharing) for a demo usage of this repo.

Alternatively, you can directly run the main script:
```shell
python3 example_racecar.py
```


<!-- LICENSE -->
## License

Distributed under the BSD 3-Clause License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Haimin Hu - [@HaiminHu](https://twitter.com/HaiminHu) - haiminh@princeton.edu


<!-- PAPER -->
## Paper

If you found this repository helpful, please consider citing one of the following papers that turned this repository into research contributions.

* iLQR-based safety filter:
```tex
@article{hu2023active,
  title={Active uncertainty reduction for safe and efficient interaction planning: A shielding-aware dual control approach},
  author={Hu, Haimin and Isele, David and Bae, Sangjae and Fisac, Jaime F},
  journal={The International Journal of Robotics Research},
  pages={02783649231215371},
  year={2023},
  publisher={SAGE Publications Sage UK: London, England}
}
```

* iLQR-based Stackelberg game solver:
```tex
@article{hu2024plays,
  title={Who Plays First? Optimizing the Order of Play in Stackelberg Games with Many Robots},
  author={Hu, Haimin and Dragotto, Gabriele and Zhang, Zixu and Liang, Kaiqu and Stellato, Bartolomeo and Fisac, Jaime F},
  journal={arXiv preprint arXiv:2402.09246},
  year={2024}
}
```

* iLQ-based game solver for data-aware planning:
```tex
@article{lidard2024blending,
  title={Blending Data-Driven Priors in Dynamic Games},
  author={Lidard, Justin and Hu, Haimin and Hancock, Asher and Zhang, Zixu and Contreras, Albert Gim{\'o} and Modi, Vikash and DeCastro, Jonathan and Gopinath, Deepak and Rosman, Guy and Leonard, Naomi and others},
  journal={arXiv preprint arXiv:2402.14174},
  year={2024}
}
```


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

This repo is inspired by the following projects:
* [Princeton Race Car](https://github.com/SafeRoboticsLab/PrincetonRaceCar_planning)
* [Ellipsoidal Toolbox (ET)](https://www.mathworks.com/matlabcentral/fileexchange/21936-ellipsoidal-toolbox-et)
* ellReach: Ellipsoidal Reachable Set Computation for Linear Time-Varying Systems (under development)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/SafeRoboticsLab/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/SafeRoboticsLab/SHARP/contributors
[forks-shield]: https://img.shields.io/github/forks/SafeRoboticsLab/repo.svg?style=for-the-badge
[forks-url]: https://github.com/SafeRoboticsLab/SHARP/network/members
[stars-shield]: https://img.shields.io/github/stars/SafeRoboticsLab/repo.svg?style=for-the-badge
[stars-url]: https://github.com/SafeRoboticsLab/SHARP/stargazers
[issues-shield]: https://img.shields.io/github/issues/SafeRoboticsLab/repo.svg?style=for-the-badge
[issues-url]: https://github.com/SafeRoboticsLab/SHARP/issues
[license-shield]: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
[license-url]: https://opensource.org/licenses/BSD-3-Clause
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/SafeRoboticsLab
[homepage-shield]: https://img.shields.io/badge/-Colab%20Notebook-orange
[homepage-url]: https://colab.research.google.com/drive/1_3HgZx7LTBw69xH61Us70xI8HISUeFA7?usp=sharing
