import numpy as np
from hypothesis import StructuralHypothesis

class SchemaEvaluator:
    def __init__(self, complexity_penalty: float = 0.05, mutation_penalty: float = 0.02):
        self.complexity_penalty = complexity_penalty
        self.mutation_penalty = mutation_penalty

    def apply_hypothesis(self, hypothesis, feature_matrix):
        if hypothesis is None or hypothesis.operator == 'null':
            return feature_matrix
        if hypothesis.operator == 'augment':
            index = hypothesis.inputs[0]
            return np.column_stack([feature_matrix, feature_matrix[:, index]])
        if hypothesis.operator == 'interaction':
            i, j = hypothesis.inputs
            interaction = feature_matrix[:, i] * feature_matrix[:, j]
            return np.column_stack([feature_matrix, interaction])
        return feature_matrix

    def fit_linear_model(self, features, targets):
        x = np.column_stack([features, np.ones(len(features))])
        try:
            weights, residuals, rank, s = np.linalg.lstsq(x, targets, rcond=None)
            if len(residuals) > 0:
                return np.mean(residuals)
            else:
                return np.mean((x @ weights - targets)**2)
        except:
            return 1e6

    def complexity(self, hypothesis):
        if hypothesis is None or hypothesis.operator == 'null': return 0.0
        if hypothesis.operator == 'augment': return self.complexity_penalty
        if hypothesis.operator == 'interaction': return self.complexity_penalty * 2
        return 1.0

    def evaluate(self, hypothesis, train_features, train_targets, validation_features, validation_targets, intervention_gain=0.0):
        baseline_error = self.fit_linear_model(validation_features, validation_targets)
        
        if hypothesis is None or hypothesis.operator == 'null':
            return 0.0
            
        candidate_validation = self.apply_hypothesis(hypothesis, validation_features)
        candidate_error = self.fit_linear_model(candidate_validation, validation_targets)
        
        compression_gain = baseline_error - candidate_error
        
        return compression_gain + intervention_gain - self.complexity(hypothesis) - self.mutation_penalty
