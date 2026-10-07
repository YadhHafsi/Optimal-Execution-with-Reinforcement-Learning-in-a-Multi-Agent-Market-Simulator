"""Order-size model used by the ABIDES background agents.

Upstream ABIDES implements this model with ``pomegranate.GeneralMixtureModel``
(version 0.14). ``pomegranate`` 0.14 is an unmaintained C extension that does
not build on current Python/numpy stacks, so this module re-implements the
*same* mixture with plain numpy:

* component 0: a log-normal with (mu, sigma) = (2.9, 1.2) on the log scale,
  weight 0.2 (small, "organic" order sizes; median about 18 shares);
* components 1..10: narrow normals centred on the round lots
  100, 200, ..., 1000 shares with standard deviation 0.15, so that after
  rounding they collapse onto the round lot itself.

The component weights are identical to upstream. Sampling draws a component
index from the weights and then a value from that component, exactly as
``GeneralMixtureModel.sample`` does, and the result is rounded to an integer
number of shares. The random stream is therefore *distributionally* identical
to upstream but not bit-for-bit identical (pomegranate consumed the numpy
RandomState differently).
"""
from typing import List

import numpy as np

# (name, parameters) of the eleven mixture components, in upstream order.
_LOGNORMAL_MU: float = 2.9
_LOGNORMAL_SIGMA: float = 1.2
_ROUND_LOTS: List[float] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
_ROUND_LOT_STD: float = 0.15

# Mixture weights, identical to the upstream pomegranate JSON specification.
_WEIGHTS: np.ndarray = np.array(
    [0.2, 0.7, 0.06, 0.004, 0.0329, 0.001, 0.0006, 0.0004, 0.0005, 0.0003, 0.0003]
)
assert abs(_WEIGHTS.sum() - 1.0) < 1e-12, "mixture weights must sum to one"


class OrderSizeModel:
    """Mixture model of background-agent order sizes (see module docstring)."""

    def __init__(self) -> None:
        self.weights: np.ndarray = _WEIGHTS / _WEIGHTS.sum()
        self.round_lots: np.ndarray = np.array(_ROUND_LOTS)

    def sample(self, random_state: np.random.RandomState) -> int:
        """Draw one order size (in shares) using the caller's ``RandomState``."""
        component = random_state.choice(len(self.weights), p=self.weights)
        if component == 0:
            size = random_state.lognormal(mean=_LOGNORMAL_MU, sigma=_LOGNORMAL_SIGMA)
        else:
            size = random_state.normal(
                loc=self.round_lots[component - 1], scale=_ROUND_LOT_STD
            )
        return int(round(size))

    def mean(self) -> float:
        """Analytical mean of the mixture (useful for tests and documentation)."""
        lognormal_mean = np.exp(_LOGNORMAL_MU + 0.5 * _LOGNORMAL_SIGMA**2)
        return float(
            self.weights[0] * lognormal_mean + np.dot(self.weights[1:], self.round_lots)
        )
