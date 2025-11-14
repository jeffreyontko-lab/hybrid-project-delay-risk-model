# Hybrid Project Delay Risk Model

This repository documents and implements a **hybrid Perceptron + Logistic Regression model** to estimate the **probability of project delay** based on key project factors such as team experience, complexity, resource availability, stakeholder engagement, dependencies, and schedule pressure.

It includes:

- A **mathematical model** (weighted linear score + sigmoid)
- A **Python implementation** (`hybrid_model.py`)
- An **Excel-based risk calculator** for non-technical users
- Documentation on **why the model was built**, **how it was calibrated**, and **how it was validated**

---

## 1. Problem Statement

Most complex IT and AI projects have a high probability of failure or delay. Industry sources (e.g., Gartner) regularly report:

- High proportions of IT projects considered unsuccessful by business sponsors
- A large percentage of AI projects failing to deliver expected value or being abandoned

However, project risk assessments are often **qualitative** (“red/amber/green”) and lack a quantitative, repeatable way to estimate the **probability of delay**.

**Goal:**  
Build a simple, interpretable model that predicts:

> **P(project delay)** as a percentage, given a small, well-defined set of project features.

---

## 2. Model Overview: Hybrid Perceptron + Logistic Regression

The model uses a **linear combination** of features (like a perceptron) and applies a **sigmoid (logistic) function** to output a probability:

1. **Linear score (z):**

\[
z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_7 x_7
\]

2. **Probability of delay:**

\[
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

3. **Risk percentage:**

\[
\text{RiskPct} = 100 \times \hat{y}
\]

This combines:

- The **interpretability** of a simple linear model  
- The **probabilistic output** of logistic regression (0–100% risk)

---

## 3. Features (Inputs)

All categorical factors use a 1–3 encoding:

- **High = 3**
- **Medium = 2**
- **Low = 1**

### 3.1 TeamExp_Avg

**Meaning:**  
Average experience level of the delivery team.

- Encoded per person: High = 3, Medium = 2, Low = 1  
- `TeamExp_Avg` is simply the average of those scores.

**Intuition:**  
Higher average experience → fewer mistakes, faster recovery → **lower risk**.

---

### 3.2 TeamExp_Var

**Meaning:**  
How uneven team experience levels are (variance).

- Low variance: everyone is similar  
- High variance: big mix of junior and senior people

**Intuition:**  
Very uneven teams can create bottlenecks (one expert helping many less experienced teammates).  
The model slightly **increases risk** when variance is high.

---

### 3.3 Complexity

**Meaning:**  
Overall difficulty of the project.

- 1 = Low complexity  
- 2 = Medium  
- 3 = High

**Intuition:**  
More complex projects have more unknowns, more integration points, and more failure modes → **higher risk**.

---

### 3.4 ResourceAvail

**Meaning:**  
Availability/commitment of key resources.

- 1 = Low availability, heavily shared  
- 2 = Partially available  
- 3 = Fully available / dedicated

**Intuition:**  
This is one of the **strongest risk reducers** in the model.  
More availability = more capacity to absorb shocks and rework → **lower risk**.

---

### 3.5 StakeholderEng

**Meaning:**  
Engagement level of business stakeholders / sponsors.

- 1 = Low engagement (conflicts, slow approvals)  
- 2 = Medium  
- 3 = High (responsive, available for testing, quick decisions)

**Intuition:**  
Low engagement → delayed approvals, unclear requirements, rework → **higher risk**.  
High engagement helps reduce risk.

---

### 3.6 Dependencies

**Meaning:**  
External dependencies and controllability.

- 1 = Low dependency risk  
- 2 = Medium  
- 3 = High (many external teams/vendors/systems)

**Intuition:**  
Dependencies are classic sources of delay (waiting on others).  
More dependency risk → **higher risk of delay**.

---

### 3.7 SchedulePressure

**Meaning:**  
How “tight” the schedule is.

Defined as:

\[
\text{SchedulePressure} = \frac{\text{Time Needed}}{\text{Time Available}}
\]

**Examples:**

- Need 3 weeks, have 3 weeks → 3/3 = 1.0  
- Need 3 weeks, have 2 weeks → 3/2 = 1.5  
- Need 2 weeks, have 4 weeks → 2/4 = 0.5

**Intuition:**  
This is the **largest single driver** in the model.  
Higher schedule pressure dramatically increases the probability of delay.

---

## 4. Weights and Final Model

Final calibrated weights:

- **Bias (intercept):** `w0 = +1.0`  
  - Chosen to reflect a **high-risk baseline environment**, consistent with reported failure rates in IT/AI projects.
- **TeamExp_Avg:** `w1 = -0.4`  
- **TeamExp_Var:** `w2 = +0.2`  
- **Complexity:** `w3 = +0.5`  
- **ResourceAvail:** `w4 = -0.8`  
- **StakeholderEng:** `w5 = -0.3`  
- **Dependencies:** `w6 = +0.7`  
- **SchedulePressure:** `w7 = +1.0`

So the full linear score is:

\[
z = 1.0
   - 0.4 \cdot \text{TeamExp\_Avg}
   + 0.2 \cdot \text{TeamExp\_Var}
   + 0.5 \cdot \text{Complexity}
   - 0.8 \cdot \text{ResourceAvail}
   - 0.3 \cdot \text{StakeholderEng}
   + 0.7 \cdot \text{Dependencies}
   + 1.0 \cdot \text{SchedulePressure}
\]

and:

\[
\text{RiskPct} = 100 \times \frac{1}{1 + e^{-z}}
\]

---

## 5. Training and Validation (Synthetic Data)

To safely experiment and validate the model structure, synthetic datasets were generated:

- **Training set:** 1,000 rows  
- **Test/validation set:** 150 rows  

Each row is a project scenario with:

- Randomized but realistic feature values  
- RiskPct computed directly from the model formula  
  (i.e., the training label is generated by the same equation)

### 5.1 Fit Metrics

Because the target label was generated *from the same model*, re-applying the model to the data yields:

- **Training R²:** 1.00  
- **Test R²:** 1.00  
- **MAE:** effectively 0 (floating-point noise)  
- **RMSE:** effectively 0

> These metrics confirm that the implementation matches the intended formula exactly.  
> They **do not** mean the model is “perfect in the real world” – only that the math is internally consistent.

---

## 6. Sensitivity Analysis

To understand which features matter most, a simple **“what-if” sensitivity analysis** was performed:

- For each feature, increase its value by +1 (where applicable)
- Hold other features constant
- Measure average change in RiskPct across many scenarios

Results (example magnitudes):

- **SchedulePressure:** largest positive impact on risk per unit increase  
- **Dependencies:** strong positive impact  
- **Complexity:** moderate positive impact  
- **ResourceAvail:** strong negative impact (risk reducer)  
- **TeamExp_Avg, StakeholderEng:** moderate negative impact  
- **TeamExp_Var:** small positive impact

This confirms qualitative expectations:

- Tight schedules + high dependencies drive risk up
- Strong staffing + engaged stakeholders drive risk down

---

## 7. Baseline Example Scenario

Example project:

- TeamExp_Avg = 2.5 (one high, one medium)  
- TeamExp_Var = 0.25  
- Complexity = 2 (medium)  
- ResourceAvail = 3 (fully available dev)  
- StakeholderEng = 1 (low engagement, frequent conflicts)  
- Dependencies = 1 (low external dependency risk)  
- SchedulePressure = 3 / 2 = 1.5  

With **bias = +1.0**, this scenario yields:

- **Baseline delay risk ≈ 63.4%**

This aligns with a high baseline failure climate and highlights that even a “decently staffed, medium-complexity project” can be risky if stakeholder engagement is low and schedule pressure is high.

---

## 8. Python Implementation

See [`hybrid_model.py`](./hybrid_model.py) for a fully self-contained implementation of:

- The logistic sigmoid
- The linear score computation
- A `predict_risk_pct(...)` helper

No external ML libraries are required.

---

## 9. Excel Risk Calculator

For non-developers, an **Excel workbook** is included:

- Sheet: `RiskModel`
- Inputs:
  - Team experience per person
  - Derived `TeamExp_Avg` and `TeamExp_Var`
  - Complexity, ResourceAvail, StakeholderEng, Dependencies
  - Time Needed, Time Available (to compute SchedulePressure)
- Output:
  - `RiskPct_%` calculated with the same formula as the Python model

This enables simple **“what-if” planning** in Excel:
- Change inputs → see updated Risk%
- Compare current vs improved scenarios
- Communicate risk transparently to stakeholders

---

## 10. Future Work

Potential next steps:

- Use real project data (historical outcomes) to re-fit weights via logistic regression
- Calibrate bias and thresholds per organization (e.g., what counts as “High risk”)
- Add Monte Carlo simulation for uncertainty in estimates
- Integrate into a portfolio dashboard or AI-assisted PM tool

---

## 11. Contact

Created by **Jeffrey Ontko** (`jeffreyontko-lab`) as an exploratory hybrid model for project delay risk prediction and explainability.

If you are interested in collaborating on:

- Applying this model to real project data  
- Extending the feature set  
- Integrating into PMO tooling  

…feel free to reach out via GitHub or LinkedIn.

