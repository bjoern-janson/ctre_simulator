import numpy as np
from dataclasses import dataclass

@dataclass
class State:
    velocity: float
    context: float

class ScalarLatentEnvironment:
    def __init__(self, z_range=(0.1, 0.3), coefficient=1.0, seed=42):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.shift_point = 2500

    def reset(self):
        self.t = 0
        return State(
            velocity=self.rng.uniform(0.5, 1.5),
            context=self.rng.uniform(0.5, 1.5)
        )

    def step(self, state: State):
        self.t += 1

        if self.t < self.shift_point:
            # Phase 1: Simple linear rule (y = 2 * v)
            y = 2.0 * state.velocity
        else:
            # Phase 2: Structural wall (y = v * context)
            y = state.velocity * state.context

        observation = {
            "velocity": state.velocity,
            "context": state.context,
            "target": y + self.rng.normal(0, 0.001)
        }

        next_state = State(
            velocity=self.rng.uniform(0.5, 1.5),
            context=self.rng.uniform(0.5, 1.5)
        )

        return observation, next_state
