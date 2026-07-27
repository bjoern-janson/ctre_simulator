import numpy as np

from hypothesis import StructuralHypothesis


class SchemaEvaluator:
    """
    Neutral evaluator for v0 CTRE.

    Scores structural hypotheses without semantic knowledge.

    Goal:
        reward improved generalization
        penalize unnecessary complexity
    """


    def __init__(
        self,
        complexity_penalty: float = 0.1,
        mutation_penalty: float = 0.05
    ):
        self.complexity_penalty = complexity_penalty
        self.mutation_penalty = mutation_penalty


    def apply_hypothesis(
        self,
        hypothesis: StructuralHypothesis,
        feature_matrix: np.ndarray
    ):
        """
        Apply a structural transformation.

        Returns an expanded feature matrix.
        """

        if hypothesis.operator == "null":
            return feature_matrix


        if hypothesis.operator == "augment":

            index = hypothesis.inputs[0]

            new_feature = (
                feature_matrix[:, index]
            )

            return np.column_stack(
                [
                    feature_matrix,
                    new_feature
                ]
            )


        if hypothesis.operator == "interaction":

            i, j = hypothesis.inputs

            interaction = (
                feature_matrix[:, i]
                *
                feature_matrix[:, j]
            )

            return np.column_stack(
                [
                    feature_matrix,
                    interaction
                ]
            )


        return feature_matrix



    def fit_linear_model(
        self,
        features: np.ndarray,
        targets: np.ndarray
    ):
        """
        Fits the best possible linear parameters
        for a given representation.

        This prevents evaluator weakness.

        The question becomes:
        is the representation sufficient?
        """

        x = np.column_stack(
            [
                features,
                np.ones(len(features))
            ]
        )

        weights, _, _, _ = np.linalg.lstsq(
            x,
            targets,
            rcond=None
        )

        predictions = x @ weights

        error = np.mean(
            np.abs(
                predictions - targets
            )
        )

        return error



    def complexity(
        self,
        hypothesis: StructuralHypothesis
    ):
        """
        Description length approximation.

        New structures have a cost.
        """

        if hypothesis.operator == "null":
            return 0.0


        if hypothesis.operator == "augment":
            return self.complexity_penalty


        if hypothesis.operator == "interaction":
            return self.complexity_penalty * 2


        return 1.0



    def evaluate(
        self,
        hypothesis: StructuralHypothesis,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
        intervention_gain: float = 0.0
    ):
        """
        Composite fitness.

        Higher is better.

        Fitness:

            validation improvement
            +
            intervention improvement
            -
            complexity cost
            -
            mutation cost

        """

        baseline_error = self.fit_linear_model(
            validation_features,
            validation_targets
        )


        candidate_train = self.apply_hypothesis(
            hypothesis,
            train_features
        )

        candidate_validation = self.apply_hypothesis(
            hypothesis,
            validation_features
        )


        candidate_error = self.fit_linear_model(
            candidate_validation,
            validation_targets
        )


        compression_gain = (
            baseline_error
            -
            candidate_error
        )


        score = (
            compression_gain
            +
            intervention_gain
            -
            self.complexity(hypothesis)
            -
            self.mutation_penalty
        )


        # Null hypothesis is special:
        # it pays no mutation cost

        if hypothesis.operator == "null":

            score = (
                -baseline_error
            )


        return score
