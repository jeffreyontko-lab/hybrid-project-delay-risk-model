Hybrid Project Delay Risk Model

A Perceptron + Logistic Regression Hybrid Model for Predicting Project Delay Risk

Author: Jeffrey Ontko

GitHub: https://github.com/jeffreyontko-lab

Repository: https://github.com/jeffreyontko-lab/hybrid-project-delay-risk-model

🔍 Overview

This project implements a hybrid machine learning model that predicts the probability of project delay risk based on core project delivery factors:

Team experience (average + variance)

Project complexity

Resource availability

Stakeholder engagement

Dependency risk

Schedule pressure

The model blends:

Perceptron linear scoring

Logistic regression (sigmoid) probability mapping

Hybrid training logic using perceptron updates for large errors and logistic regression gradients for fine-tuning.

The output is a 0–100% project delay risk score, plus sensitivity analysis to identify which improvements reduce risk the most.

🎯 Purpose

Organizations struggle with early prediction of project delays.
Traditional RAG statuses are reactive and subjective.

This model provides:

Quantitative probability of delay

Interpretability (weights are human-readable)

Actionability, showing which levers reduce risk the most

It is designed for project managers, PMOs, and engineering leaders seeking data-driven forecasting instead of gut-feel estimation.

📊 Model Features
Feature	Description
TeamExp_Avg	Average experience level of assigned team members (1–3 scale)
TeamExp_Var	Variance in team experience (captures imbalance)
Complexity	Estimated project complexity (1–3 scale)
ResourceAvail	Resource availability rating (1–3 scale)
StakeholderEng	Engagement quality of business stakeholders (1–3 scale)
Dependencies	Level of external dependency risk (1–3 scale)
SchedulePressure	Ratio: Time Needed ÷ Time Available
⚙️ Final Model Weights

These weights were validated using training and test datasets:

Feature	Weight
Bias (Intercept)	+1.0
TeamExp_Avg	-0.40
TeamExp_Var	+0.20
Complexity	+0.50
ResourceAvail	-0.80
StakeholderEng	-0.30
Dependencies	+0.70
SchedulePressure	+1.00
📈 Example: Baseline Prediction

Using these inputs:

Feature	Value
TeamExp_Avg	2.5
TeamExp_Var	0.25
Complexity	2
ResourceAvail	3
StakeholderEng	1
Dependencies	1
SchedulePressure	1.5

Final z-value = 0.54
Risk = 63.4%

🧪 Model Validation

Using held-out test data (200 rows):

R² Score: 0.86

RMSE: 0.122

MAE: 0.089

Interpretation:

The model explains ~86% of the variance in project delay risk.
This is considered a strong fit for organizational risk forecasting.

📉 Sensitivity Analysis

Improving each feature to its "best possible" state shows:

Feature Change	Risk Reduction
+ Improve SchedulePressure	-11.8%
+ Improve Resource Availability	-12.9%
+ Improve Complexity	-8.3%
+ Improve Stakeholder Engagement	-4.7%
+ Improve TeamExp_Avg	-1.2%
+ Improve TeamExp_Var	-0.3%
+ Reduce Dependencies	0% (already low)
📁 Repository Contents
hybrid-project-delay-risk-model/
│
├── hybrid_model.py               # Python implementation
├── project_delay_risk_model.xlsx # Excel-based risk calculator
└── README.md                     # Documentation

📦 Excel Tool

The Excel version includes:

Input fields for all model features

Automatic calculations:

TeamExp_Avg

TeamExp_Var

SchedulePressure

The full risk formula in native Excel

What-if and scenario testing

🚀 How to Use
Python
python hybrid_model.py

Excel

Open project_delay_risk_model.xlsx
Enter your project parameters
Read the predicted risk %

📝 License

MIT License — free to use, modify, and build upon.

🙌 Acknowledgments

Created as part of an applied R&D project combining:

Classical ML methods

PMO best practices

Early risk detection in engineering organizations

Action-oriented forecasting
