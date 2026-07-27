from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class StructuralHypothesis:
    """
    Anonymous compiler-IR representation change.

    No semantic labels.
    No environment knowledge.

    The hypothesis only describes:
    - what operation to apply
    - which numeric inputs are involved
    - what mathematical transform is introduced
    """

    operator: str
    inputs: tuple
    transform: str
    parameters: tuple = field(default_factory=tuple)



class AnonymousHypothesisGenerator:
    """
    Generates candidate representation changes purely
    from residual geometry.

    Available v0 operators:

    null:
        keep current representation

    augment:
        add a transformed copy of one feature

    interaction:
        add a product feature between two inputs
    """


    @staticmethod
    def generate(
        residual_history: np.ndarray,
        feature_matrix: np.ndarray,
    ) -> List[StructuralHypothesis]:

        hypotheses = []


        # -------------------------------------------------
        # 1. Null hypothesis
        # -------------------------------------------------
        # Mandatory:
        # evolution is allowed to choose stasis

        hypotheses.append(
            StructuralHypothesis(
                operator="null",
                inputs=tuple(),
                transform="none"
            )
        )


        n_features = feature_matrix.shape[1]


        # -------------------------------------------------
        # 2. Single feature residual scaling
        # -------------------------------------------------

        residual_signal = np.abs(
            residual_history
        )


        for i in range(n_features):

            feature = feature_matrix[:, i]

            if np.std(feature) == 0:
                continue


            correlation = np.corrcoef(
                residual_signal,
                np.abs(feature)
            )[0, 1]


            if (
                not np.isnan(correlation)
                and abs(correlation) > 0.35
            ):

                hypotheses.append(
                    StructuralHypothesis(
                        operator="augment",
                        inputs=(i,),
                        transform="residual_scale"
                    )
                )


        # -------------------------------------------------
        # 3. Interaction search
        # -------------------------------------------------

        if n_features >= 2:

            mean_error = np.mean(
                residual_signal
            )

            if mean_error > 0.05:

                for i in range(n_features):
                    for j in range(i + 1, n_features):

                        hypotheses.append(
                            StructuralHypothesis(
                                operator="interaction",
                                inputs=(i, j),
                                transform="product"
                            )
                        )


        return hypotheses
