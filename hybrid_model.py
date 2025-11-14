"""
Hybrid Project Delay Risk Model

Implements a hybrid Perceptron + Logistic Regression style model
to estimate the probability of project delay based on:

- TeamExp_Avg
- TeamExp_Var
- Complexity
- ResourceAvail
- StakeholderEng
- Dependencies
- SchedulePressure

Bias is set to +1.0 to reflect a relatively high-risk baseline environment.
"""

from math import exp


# Final calibrated weights
WEIGHTS = {
    "bias": 1.0,              # Intercept / baseline offset
    "teamexp_avg": -0.4,
    "teamexp_var": 0.2,
    "complexity": 0.5,
    "resource_avail": -0.8,   # Higher availability reduces risk
    "stakeholder_eng": -0.3,
    "dependencies": 0.7,
    "schedule_pressure": 1.0,
}


def sigmoid(z: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + exp(-z))


def compute_z(
    teamexp_avg: float,
    teamexp_var: float,
    complexity: int,
    resource_avail: int,
    stakeholder_eng: int,
    dependencies: int,
    schedule_pressure: float,
    weights: dict = WEIGHTS,
) -> float:
    """
    Compute the linear score z = w0 + Σ(w_i * x_i).
    """
    z = (
        weights["bias"]
        + weights["teamexp_avg"] * teamexp_avg
        + weights["teamexp_var"] * teamexp_var
        + weights["complexity"] * complexity
        + weights["resource_avail"] * resource_avail
        + weights["stakeholder_eng"] * stakeholder_eng
        + weights["dependencies"] * dependencies
        + weights["schedule_pressure"] * schedule_pressure
    )
    return z


def predict_risk_pct(
    teamexp_avg: float,
    teamexp_var: float,
    complexity: int,
    resource_avail: int,
    stakeholder_eng: int,
    dependencies: int,
    schedule_pressure: float,
    weights: dict = WEIGHTS,
) -> float:
    """
    Predict risk as a percentage (0–100).
    """
    z = compute_z(
        teamexp_avg=teamexp_avg,
        teamexp_var=teamexp_var,
        complexity=complexity,
        resource_avail=resource_avail,
        stakeholder_eng=stakeholder_eng,
        dependencies=dependencies,
        schedule_pressure=schedule_pressure,
        weights=weights,
    )
    return 100.0 * sigmoid(z)


if __name__ == "__main__":
    # Example baseline scenario (one high, one medium experience)
    member1 = 3
    member2 = 2

    teamexp_avg = (member1 + member2) / 2.0
    teamexp_var = ((member1 - teamexp_avg) ** 2 + (member2 - teamexp_avg) ** 2) / 2.0

    complexity = 2          # Medium
    resource_avail = 3      # Fully available
    stakeholder_eng = 1     # Low
    dependencies = 1        # Low
    schedule_pressure = 3 / 2.0  # Time needed / Time available = 1.5

    risk = predict_risk_pct(
        teamexp_avg=teamexp_avg,
        teamexp_var=teamexp_var,
        complexity=complexity,
        resource_avail=resource_avail,
        stakeholder_eng=stakeholder_eng,
        dependencies=dependencies,
        schedule_pressure=schedule_pressure,
    )

    print(f"Baseline delay risk (bias=+1.0): {risk:.2f}%")
