import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from hypothesis import AnonymousHypothesisGenerator

@dataclass
class TorsionTracker:
    residuals: List[float] = field(default_factory=list)
    window_size: int = 100

    def update(self, error, improved=False):
        self.residuals.append(abs(error))
        if len(self.residuals) > self.window_size:
            self.residuals.pop(0)

    def torsion(self):
        if len(self.residuals) < 20: return 0.0
        return np.mean(self.residuals)

    def reset(self):
        self.residuals = []

@dataclass
class REEAgent:
    learning_rate: float = 0.01
    torsion_threshold: float = 0.01
    weights: np.ndarray = field(default_factory=lambda: np.zeros(2))
    history: List[Dict] = field(default_factory=list)
    torsion_tracker: TorsionTracker = field(default_factory=TorsionTracker)
    active_hypothesis: Optional[any] = None
    mutation_count: int = 0

    def features(self, observation):
        v = observation.get('velocity', 0.0)
        c = observation.get('context', 0.0)
        basis = [1.0, v]
        if self.active_hypothesis and self.active_hypothesis.operator == 'interaction':
            basis.append(v * c)
        return np.array(basis)

    def predict(self, observation):
        x = self.features(observation)
        if len(self.weights) != len(x):
            self.weights = np.resize(self.weights, len(x))
        return np.dot(self.weights, x)

    def representation_size(self):
        return len(self.weights)

    def should_sleep(self):
        return self.torsion_tracker.torsion() > self.torsion_threshold

    def commit_hypothesis(self, hypothesis):
        if hypothesis.operator == 'null': return
        current_op = getattr(self.active_hypothesis, 'operator', None) if self.active_hypothesis else None
        if hypothesis.operator == current_op: return

        print(f"[ORCHESTRATION] Committing: {hypothesis.operator}")
        self.active_hypothesis = hypothesis
        self.mutation_count += 1
        self.torsion_tracker.reset()

    def sleep_cycle(self, observations, errors):
        feature_matrix = np.array([[o['velocity'], o['context']] for o in observations])
        candidates = AnonymousHypothesisGenerator.generate(np.array(errors), feature_matrix)
        
        scored_candidates = []
        for h in candidates:
            score = 0.0
            if h.operator == 'null':
                score = -np.mean(np.array(errors)**2)
            elif h.operator == 'interaction':
                vc = feature_matrix[:, 0] * feature_matrix[:, 1]
                # Simple correlation as a fitness proxy
                score = np.abs(np.corrcoef(vc, errors)[0, 1]) if np.std(vc) > 1e-6 else 0.0
            
            scored_candidates.append((h, score))
        return scored_candidates

    def update(self, observation, target):
        x = self.features(observation)
        if len(self.weights) != len(x):
            self.weights = np.resize(self.weights, len(x))

        prediction = np.dot(self.weights, x)
        error = target - prediction
        old_abs = abs(error)
        self.weights += self.learning_rate * error * x

        new_abs = abs(target - np.dot(self.weights, x))
        self.torsion_tracker.update(error, improved=(new_abs < old_abs))

        self.history.append({
            'observation': observation, 'prediction': float(prediction),
            'target': float(target), 'error': float(error),
            'torsion': self.torsion_tracker.torsion(), 'mutation_count': self.mutation_count
        })

        if self.should_sleep():
            window = self.history[-30:]
            obs_list = [h.get('observation') for h in window if h.get('observation') is not None]
            err_list = [h.get('error') for h in window if h.get('observation') is not None]
            if len(obs_list) > 10:
                hyps = self.sleep_cycle(obs_list, err_list)
                if hyps:
                    # Sort by score (index 1 of the tuple)
                    best_hyp, best_score = max(hyps, key=lambda x: x[1])
                    if best_hyp.operator != 'null' and self.mutation_count == 0:
                        self.commit_hypothesis(best_hyp)
        return float(error)
