from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List

@dataclass
class RLState:
    """Everything the policy can observe before choosing an action."""
    hypothesis_embedding_dim: int = 384        # matches all-MiniLM-L6-v2
    evidence_count: int = 0
    peer_reviewed_ratio: float = 0.0
    plan_task_type: str = "classification"
    baseline_score: float = 0.0                # last known baseline metric
    budget_remaining: float = 1.0              # normalized 0..1
    prior_attempts: int = 0
    prior_failures: int = 0

@dataclass
class RLAction:
    """Discrete experiment configuration the policy can choose."""
    model_family: str = "RandomForest"         # e.g. RF, GBM, LogisticRegression, MLP
    preprocessing: str = "standard"            # standard | minmax | none
    feature_engineering: str = "none"          # none | pca | polynomial
    retry_budget: int = 2                      # 0..4
    training_epochs: int = 0                   # for MLP

@dataclass
class RLReward:
    """
    reward = w_metric * normalized_metric
           + w_validity * is_valid
           - w_failure * failure_ratio
           - w_cost * normalized_cost
    """
    normalized_metric: float = 0.0             # e.g. F1 / 1.0
    is_valid: int = 0                          # 1 if run ok
    failure_ratio: float = 0.0                 # failures / attempts
    normalized_cost: float = 0.0               # runtime / budget
    weights: Dict[str, float] = field(default_factory=lambda: {
        "metric": 1.0, "validity": 0.5, "failure": 0.5, "cost": 0.2,
    })

    def value(self) -> float:
        w = self.weights
        return (w["metric"] * self.normalized_metric
                + w["validity"] * self.is_valid
                - w["failure"] * self.failure_ratio
                - w["cost"] * self.normalized_cost)

def transition(state: RLState, action: RLAction, result: Dict[str, Any],
               attempts: int, duration_s: float, budget_s: float) -> RLReward:
    """Build a reward from a completed experiment run."""
    metrics = result.get("metrics", {}) or {}
    metric_val = float(
        metrics.get("f1", metrics.get("r2", metrics.get("accuracy", 0.0)))
    )
    return RLReward(
        normalized_metric=max(0.0, min(1.0, metric_val)),
        is_valid=1 if result.get("ok") else 0,
        failure_ratio=1.0 - (1.0 / max(1, attempts)),
        normalized_cost=min(1.0, duration_s / max(1.0, budget_s)),
    )