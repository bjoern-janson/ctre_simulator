import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from hypothesis import AnonymousHypothesisGenerator


@dataclass
class TorsionTracker:
    """
    Tracks persistent representation pressure.

    High torsion means:
    local parameter repair is no longer
    reducing prediction error efficiently.
    """

    residuals: List[float] = field(default_factory=list)
    repair_attempts: int = 0
    repair_successes: int = 0
    window: int = 50


    def update(self, error: float, improved: bool):

        self.residuals.append(abs(error))

        self.repair_attempts += 1

        if improved:
            self.repair_successes += 1


        if len(self.residuals) > self.window:
            self.residuals.pop(0)


    def torsion(self):
        """
        R_torsion =
        contradiction pressure /
        local repair efficiency
        """

        if not self.residuals:
            return 0.0


        contradiction_pressure = np.mean(
            self.residuals
        )


        repair_efficiency = (
            self.repair_successes
            /
            max(self.repair_attempts, 1)
        )


        return (
            contradiction_pressure
            /
            max(repair_efficiency, 1e-6)
        )


    def reset(self):

        self.residuals.clear()
        self.repair_attempts = 0
        self.repair_successes = 0



@dataclass
class REEAgent:
    """
    Agent E: Representation Evolution Engine.

    Responsibilities:

    - state adaptation
    - torsion monitoring
    - sleep trigger
    - hypothesis generation
    """

    learning_rate: float = 0.05
    torsion_threshold: float = 0.5


    weights: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0])
    )


    torsion_tracker: TorsionTracker = field(
        default_factory=TorsionTracker
    )


    active_hypothesis: Optional[object] = None

    mutation_count: int = 0


    history: List[Dict] = field(
        default_factory=list
    )


    def predict(self, observation):

        features = self.features(
            observation
        )

        return float(
            np.dot(
                self.weights,
                features
            )
        )


    def features(self, observation):

        features = [
            observation["velocity"],
            observation["context"]
        ]


        if (
            self.active_hypothesis
            and self.active_hypothesis.operator == "augment"
        ):

            index = self.active_hypothesis.inputs[0]

            features.append(
                features[index]
            )


        if (
            self.active_hypothesis
            and self.active_hypothesis.operator == "interaction"
        ):

            i, j = self.active_hypothesis.inputs

            features.append(
                features[i] * features[j]
            )


        return np.array(features)



    def update(self, observation, target):

        x = self.features(
            observation
        )


        if len(self.weights) != len(x):

            self.weights = np.resize(
                self.weights,
                len(x)
            )


        prediction = np.dot(
            self.weights,
            x
        )


        error = target - prediction

        old_error = abs(error)


        self.weights += (
            self.learning_rate
            *
            error
            *
            x
        )


        new_prediction = np.dot(
            self.weights,
            x
        )


        new_error = abs(
            target - new_prediction
        )


        self.torsion_tracker.update(
            error,
            improved=(new_error < old_error)
        )


        self.history.append(
            {
                "prediction": float(prediction),
                "target": float(target),
                "error": float(error),
                "torsion": self.torsion_tracker.torsion(),
                "mutation_count": self.mutation_count
            }
        )


        return float(error)



    def should_sleep(self):

        current_torsion = self.torsion_tracker.torsion()

        # print(
            # "torsion:",
            # round(current_torsion, 4)
        # )


        return (
            current_torsion
            >
            self.torsion_threshold
        )



    def sleep_cycle(self, observations, errors):

        feature_matrix = np.array(
            [
                [
                    o["velocity"],
                    o["context"]
                ]
                for o in observations
            ]
        )


        hypotheses = AnonymousHypothesisGenerator.generate(
            np.array(errors),
            feature_matrix
        )


        return hypotheses



    def commit_hypothesis(self, hypothesis):

        if hypothesis.operator == "null":
            return


        self.active_hypothesis = hypothesis

        self.mutation_count += 1


        feature_count = len(
            self.features(
                {
                    "velocity": 1.0,
                    "context": 1.0
                }
            )
        )


        if len(self.weights) < feature_count:

            self.weights = np.resize(
                self.weights,
                feature_count
            )


        self.torsion_tracker.reset()



    def representation_size(self):

        return len(self.weights)
