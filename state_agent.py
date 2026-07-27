import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class StateAgent:
    """
    Agent B: State-only adaptation baseline.

    Fixed representation:
        y_hat = w * velocity + b

    Learns parameters continuously but cannot modify
    the underlying representation.
    """

    learning_rate: float = 0.05

    weights: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0])
    )

    history: List[Dict] = field(
        default_factory=list
    )


    def predict(self, observation):
        """
        Fixed linear representation.

        Features:
            [velocity, context]
        """

        x = np.array([
            observation["velocity"],
            observation["context"]
        ])

        return float(
            np.dot(self.weights, x)
        )


    def update(self, observation, target):
        """
        Parameter adaptation.

        This is intentionally strong:
        the agent receives gradient information
        and continuously optimizes parameters.

        It cannot add new features.
        """

        x = np.array([
            observation["velocity"],
            observation["context"]
        ])

        prediction = np.dot(
            self.weights,
            x
        )

        error = target - prediction

        gradient = (
            error
            * x
        )

        self.weights += (
            self.learning_rate
            * gradient
        )

        self.history.append(
            {
                "prediction": float(prediction),
                "target": float(target),
                "error": float(error),
                "weights": self.weights.copy()
            }
        )

        return float(error)


    def representation_size(self):
        """
        Used for compute parity comparisons.
        """

        return len(self.weights)


    def mutation_count(self):
        """
        State agent never changes representation.
        """

        return 0


    def reset_history(self):
        self.history.clear()
