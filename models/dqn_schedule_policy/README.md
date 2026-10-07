# Released DQN execution policy (learning configuration)

Schedule-mode policy of `learn.yaml`: it chooses a participation multiplier of the TWAP rate every 10 s inside a completion corridor. It completes every order and matches TWAP's mean shortfall (200 seeds: -9.43 against -9.50, paired difference +0.07 cents per share, p = 0.88); see docs/CHANGES.md C11.

Trained with `experiments/train_dqn.py` (run `lr_5e-4_seed_10` of `experiments/configs/learn.yaml`, see docs/CHANGES.md C11).

## Training settings

| Setting | Value |
|---|---|
| Configuration | `experiments/configs/learn.yaml` |
| Terminal penalty (never triggers: the corridor completes the order) | 5 / 5 |
| Initial learning rate (linear decay to 0) | 0.0005 over 90000 steps |
| Exploration | epsilon 1.0 to 0.02 over 20000 steps |
| Discount | 0.99 |
| Q-head hidden layers | [50, 20] |
| Environment steps | 100000 |
| Seed | 10 |

Files: `checkpoint/` (RLlib 2.2.0 checkpoint), `run_config.json`,
`progress.csv` and `episodes.csv` (training logs), `evaluation/` (out-of-sample
episodes per scenario).

## Out-of-sample evaluation

Evaluation seeds 1000 + i, one execution window each; statistics of the
normalised implementation shortfall (episode total, cents per share of the
parent order), terminal penalty, fraction of the window used and completion.

| scenario | n | E(IS total) | SD(IS total) | E(IS step mean) | E(Pen) | E(T) | E(executed) | P(complete) |
|---|---|---|---|---|---|---|---|---|
| momentum_24 | 50 | -7.074 | 27.22 | -0.0393 | 0 | 1 | 20000 | 1 |
| momentum_6 | 50 | -8.246 | 19.43 | -0.04581 | 0 | 1 | 20000 | 1 |
| noise_10 | 50 | -7.33 | 28.32 | -0.04072 | 0 | 1 | 20000 | 1 |
| noise_2000 | 50 | -14.38 | 22.47 | -0.0799 | 0 | 1 | 20000 | 1 |
| standard | 50 | -5.564 | 19.41 | -0.03091 | 0 | 1 | 20000 | 1 |

## Using the policy

```bash
python -m experiments.evaluate --config experiments/configs/learn.yaml --policies RL \
    --checkpoint models/dqn_schedule_policy
```

In Python:

```python
from experiments.policies import RLlibDQNPolicy
from experiments.common import env_kwargs_from_config, load_config

cfg = load_config("experiments/configs/learn.yaml")
env_kwargs = env_kwargs_from_config(cfg)
env_kwargs.update({k: v for k, v in dict(beta_not_enough=5, beta_too_much=5).items() if v != env_kwargs.get(k)})
policy = RLlibDQNPolicy("models/dqn_schedule_policy", env_config=env_kwargs)
action = policy.get_action(state)      # state: the 6-feature observation of learn.yaml
q_values = policy.q_values(state)
```
