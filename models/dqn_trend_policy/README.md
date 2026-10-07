# Released DQN execution policy (trending-market configuration)

Policy of `learn_trend.yaml` (market with a random-sign drift of 0.04 cents per second; schedule mode, 10 s decisions, flat corridor 0.2). It is the **median** of five training runs by out-of-sample gain over TWAP, released instead of the best run to avoid selection bias. On 200 fresh seeds in the trending market it gains +5.68 cents per share over TWAP (se 1.37, p < 0.001) and completes every order. In the original market without drift it loses 6.0 cents per share to TWAP (p < 0.001): it follows price moves, which pays only when they persist. Of the five runs, three gain +5.7 to +7.0 and two (learning rate 1e-3 and 5e-4 with seed 10) gain nothing (-2.2 and -2.4, not significant). See docs/CHANGES.md C12.

Trained with `experiments/train_dqn.py` (run `lr_5e-4_seed_20` of `experiments/configs/learn_trend.yaml`, see docs/CHANGES.md C12).

## Training settings

| Setting | Value |
|---|---|
| Configuration | `experiments/configs/learn_trend.yaml` |
| Terminal penalty (per share not executed / in excess) | 5 / 5 |
| Initial learning rate (linear decay to 0) | 0.0005 over 135000 steps |
| Exploration | epsilon 1.0 to 0.02 over 30000 steps |
| Discount | 0.99 |
| Q-head hidden layers | [50, 20] |
| Environment steps | 150000 |
| Seed | 20 |

Files: `checkpoint/` (RLlib 2.2.0 checkpoint), `run_config.json`,
`progress.csv` and `episodes.csv` (training logs), `evaluation/` (out-of-sample
episodes per scenario).

## Out-of-sample evaluation

Evaluation seeds 1000 + i, one execution window each; statistics of the
normalised implementation shortfall (episode total, cents per share of the
parent order), terminal penalty, fraction of the window used and completion.

| scenario | n | E(IS total) | SD(IS total) | E(IS step mean) | E(Pen) | E(T) | E(executed) | P(complete) |
|---|---|---|---|---|---|---|---|---|
| no_drift | 200 | -14.76 | 25.74 | -0.1042 | 0 | 0.8772 | 20000 | 1 |
| standard | 200 | -0.5564 | 46.02 | -0.02837 | 0 | 0.896 | 20000 | 1 |

## Using the policy

```bash
python -m experiments.evaluate --config experiments/configs/learn_trend.yaml --policies RL \
    --checkpoint models/dqn_trend_policy 
```

In Python:

```python
from experiments.policies import RLlibDQNPolicy
from experiments.common import env_kwargs_from_config, load_config

cfg = load_config("experiments/configs/learn_trend.yaml")
env_kwargs = env_kwargs_from_config(cfg)
env_kwargs.update({k: v for k, v in dict(beta_not_enough=5, beta_too_much=5).items() if v != env_kwargs.get(k)})
policy = RLlibDQNPolicy("models/dqn_trend_policy", env_config=env_kwargs)
action = policy.get_action(state)      # state: the 8-feature observation of markets-execution-v0
q_values = policy.q_values(state)
```
