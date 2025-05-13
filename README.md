# DARRL
This repository is the implementation of our research "**[Trustworthy Autonomous Driving via Defense-Aware Robust Reinforcement Learning against Worst-Case Observational Perturbations](https://www.researchgate.net/publication/381075887_Trustworthy_autonomous_driving_via_defense-aware_robust_reinforcement_learning_against_worst-case_observational_perturbations)**". This work has been published in *Transportation Research Part C: Emerging Technologies*. 

## Introduction
### Illustration of the proposed DARRL framework for trustworthy autonomous driving
<img src="/framework.png" alt="ENV" width="777" height="366">
Despite the substantial advancements in reinforcement learning (RL) in recent years, ensuring trustworthiness remains a formidable challenge when applying this technology to safety-critical autonomous driving domains. One pivotal bottleneck is that well-trained driving policy models may be particularly vulnerable to observational perturbations or perceptual uncertainties, potentially leading to severe failures. In view of this, we present a novel defense-aware robust RL approach tailored for ensuring the robustness and safety of autonomous vehicles in the face of worst-case attacks on observations. The proposed paradigm primarily comprises two crucial modules: an adversarial attacker and a robust defender. Specifically, the adversarial attacker is devised to approximate the worst-case observational perturbations that attempt to induce safety violations (e.g., collisions) in the RL-driven autonomous vehicle. Additionally, the robust defender is developed to facilitate the safe RL agent to learn robust optimal policies that maximize the return while constraining the policy and cost perturbed by the adversarial attacker within specified bounds. Finally, the proposed technique is assessed across three distinct traffic scenarios: highway, on-ramp, and intersection. The simulation and experimental results indicate that our scheme enables the agent to execute trustworthy driving policies, even in the presence of the worst-case observational perturbations.

## User Guidance
### Installation
This repo is developed using Python 3.7 and PyTorch 1.3.1+CPU in Ubuntu 16.04. 

We utilize the proposed DARRL approach to train the autonomous driving agent in the popular [Simulation of Urban Mobility](https://eclipse.dev/sumo/) (SUMO, Version 1.2.0) platform.

We believe that our code can also run on other operating systems with different versions of Python, PyTorch and SUMO, but we have not verified it.

The required packages can be installed using

	pip install -r requirements.txt

Additionally, we have verified that the code remains effective under the SUMO 1.22.0, Python 3.8.18, and PyTorch 1.8.0 environments.

### Run
 Users can leverage the following command to run the code in the terminal and train the autonomous driving agent in traffic flows:

	python Main.py



## Acknowledgement
We greatly appreciate the important references provided by the code repository [BO](https://github.com/bayesian-optimization/BayesianOptimization) for the implementation of our research.

## Citation
If you find this repository helpful for your research, we would greatly appreciate it if you could star our repository and cite our work.
```
@article{HE2024104632,
title = {Trustworthy autonomous driving via defense-aware robust reinforcement learning against worst-case observational perturbations},
journal = {Transportation Research Part C: Emerging Technologies},
volume = {163},
pages = {104632},
year = {2024},
issn = {0968-090X},
doi = {https://doi.org/10.1016/j.trc.2024.104632},
url = {https://www.sciencedirect.com/science/article/pii/S0968090X24001530},
author = {Xiangkun He and Wenhui Huang and Chen Lv},
keywords = {Autonomous vehicle, Traffic safety, Robust decision making, Reinforcement learning, Trustworthy AI},
}
```
