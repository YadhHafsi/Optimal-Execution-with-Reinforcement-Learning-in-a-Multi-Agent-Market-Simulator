# Released DQN execution policy

Trained with `experiments/train_campaign.py` (run `beta_100_lr_0.001_seed_10`, rank 1 of
9 in the campaign ranking: completion rate first, then mean
implementation shortfall).

## Training settings

| Setting | Value |
|---|---|
| Configuration | `experiments/configs/paper.yaml` |
| Terminal penalty (per share not executed / in excess) | 100 / 100 |
| Initial learning rate (linear decay to 0) | 0.001 over 900000 steps |
| Exploration | epsilon 1.0 to 0.02 over 100000 steps |
| Discount | 0.9999 |
| Q-head hidden layers | [50, 20] |
| Environment steps | 1000000 |
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
| momentum_24 | 50 | -11.29 | 19.05 | -0.01129 | 0 | 0.5556 | 20000 | 1 |
| momentum_6 | 50 | -16.98 | 19.65 | -0.01698 | 0 | 0.5556 | 20000 | 1 |
| noise_10 | 50 | -15.44 | 18.71 | -0.01544 | 0 | 0.5556 | 20000 | 1 |
| noise_2000 | 50 | -15.67 | 17.99 | -0.01567 | 0 | 0.5556 | 20000 | 1 |
| standard | 50 | -17.74 | 17.19 | -0.01774 | 0 | 0.5556 | 20000 | 1 |

## Using the policy

```bash
python -m experiments.evaluate --config experiments/configs/paper.yaml --policies RL \
    --checkpoint models/dqn_execution_policy --env-override beta_not_enough=100 --env-override beta_too_much=100
```

In Python:

```python
from experiments.policies import RLlibDQNPolicy
from experiments.common import env_kwargs_from_config, load_config

cfg = load_config("experiments/configs/paper.yaml")
env_kwargs = env_kwargs_from_config(cfg)
env_kwargs.update(beta_not_enough=100, beta_too_much=100)
policy = RLlibDQNPolicy("models/dqn_execution_policy", env_config=env_kwargs)
action = policy.get_action(state)      # state: the 8-feature observation of markets-execution-v0
q_values = policy.q_values(state)
```
