import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class State:
    velocity: float
    context: float


class ScalarLatentEnvironment:
    """
    v0 CTRE environment.

    Hidden rule:
        y = a * v * (1 + z)

    Observable:
        velocity
        context

    Hidden:
        z

    The agent never receives z.
    """

    def __init__(
        self,
        z_range=(0.1, 0.3),
        coefficient=1.0,
        seed=42
    ):
        self.z_range = z_range
        self.coefficient = coefficient
        self.rng = np.random.default_rng(seed)

        self.z = None
        self.t = 0


    def reset(self):
        """
        Reset environment state.
        Hidden z is sampled but never exposed.
        """

        self.t = 0
        self.z = self.rng.uniform(
            self.z_range[0],
            self.z_range[1]
        )

        return State(
            velocity=1.0,
            context=1.0
        )


    def step(self, state: State):
        """
        Apply hidden transition.

        Agent observes:
            input features

        Agent receives:
            output y

        Agent does NOT receive:
            z
        """

        self.t += 1

        y = (
            self.coefficient
            * state.velocity
            * (1.0 + self.z)
        )

        observation = {
            "velocity": state.velocity,
            "context": state.context,
            "target": y
        }

        # evaluation only
        metadata = {
            "hidden_z": self.z
        }

        return observation, metadata


    def intervention(self, velocity: float):
        """
        do(velocity=x)

        Used only for causal evaluation.
        """

        return (
            self.coefficient
            * velocity
            * (1.0 + self.z)
        )


    def shift_latent(self, z_range):
        """
        Change environment distribution.
        Used for transfer evaluation.
        """

        self.z_range = z_range
