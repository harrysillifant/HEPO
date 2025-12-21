# HEPO

---

## Overview

Standard reinforcement learning methods rely solely on task rewards, which can lead to slow convergence or unstable training when rewards are sparse. HEPO augments policy optimization by explicitly incorporating heuristic rewards as auxiliary guidance while preserving the original task objective. This framework enables systematic investigation of heuristic reward design and its effect on policy learning.

The implementation is designed for clarity and extensibility rather than production use.

---

## Method

HEPO follows an on-policy training paradigm similar to PPO. During rollouts, both task rewards and heuristic rewards are collected. These signals are processed separately and combined within the policy optimization procedure to guide learning without directly replacing the task reward. This separation allows heuristic guidance to shape exploration and early learning while maintaining alignment with the true objective.

---

## Installation

```bash
git clone https://github.com/hacosi/hepo.git
cd hepo 
pip install -r requirements.txt
```


---

## Usage

```python
from hepo import HEPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("LunarLander-v3", n_envs=4)

model = HEPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
)

model.learn(total_timesteps=1_000_000)
```

