# 🏢 Employee Churn Prediction
A machine learning project to predict employee attrition, helping organizations identify at-risk employees and take proactive retention measures.

---

## 📋 Table of Contents

- [What is Employee Churn?](#-what-is-employee-churn)
- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Strategic Recommendations](#-strategic-recommendations)

---

## ❓ What is Employee Churn?

**Employee churn** (also known as employee turnover or attrition) refers to employees leaving an organization, whether voluntarily or involuntarily. High churn rates can significantly impact a company through:

- 💰 Increased recruitment and training costs
- 📉 Loss of institutional knowledge and productivity
- 👥 Decreased team morale and engagement

This project uses machine learning to predict which employees are likely to leave, enabling HR teams to intervene early and improve retention strategies.

---

## 🎯 Project Overview

### Objective

Predict whether an employee will leave the company based on various factors such as job satisfaction, workload, tenure, and performance metrics.

### Approach

| Component | Description |
|-----------|-------------|
| **Algorithm** | Random Forest Classifier |
| **Imbalance Handling** | Class weighting + Stratified K-Fold Cross-Validation |
| **Feature Selection** | MDI (Mean Decrease in Impurity) |
| **Validation Strategy** | 5-Fold Stratified Cross-Validation |
| **Deployment** | Streamlit Web Application |

---

## 🚀 Demo

Try the live prediction model here:

[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://employee-turnover-predictor.streamlit.app/) 

Enter employee details and get instant predictions on their likelihood of leaving!

---

## 📁 Project Structure

```
employee-churn-prediction/
├── Data/
│   └── HR_comma_sep.csv          # Dataset
├── Notebook/
│   └── Employee_Churn_Prediction.ipynb      # Jupyter notebook with full analysis
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 📊 Dataset

### Overview

| Metric | Value |
|--------|-------|
| **Total Samples** | 11,991 (after removing duplicates) |
| **Stayed (Class 0)** | 10,000 (83.40%) |
| **Left (Class 1)** | 1,991 (16.60%) |

### Features

| Feature | Description |
|---------|-------------|
| `satisfaction_level` | Employee's job satisfaction score (0-1) |
| `last_evaluation` | Score from most recent performance evaluation |
| `number_project` | Number of projects handled/completed |
| `average_monthly_hours` | Average hours worked per month |
| `time_spend_company` | Years spent at the company |
| `Work_accident` | Whether employee had a workplace accident (0/1) |
| `promotion_last_5years` | Whether promoted in last 5 years (0/1) |
| `Department` | Department where employee works |
| `salary` | Salary level (low/medium/high) |

### Target Variable

- **`left`**: Binary indicator (1 = Left, 0 = Stayed)

### Data Source

The dataset used is the **HR Analytics Dataset** from Kaggle.

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/giripujar/hr-analytics)

---

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction.git
   cd Employee-Churn-Prediction
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Run the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Explore the Analysis Notebook

```bash
jupyter notebook Notebook/Employee_Churn_Prediction.ipynb
```

---

## 🔬 Methodology

### 1. Data Preparation

- **Train-Test Split:** 90% training, 10% testing
- **Stratified splitting** to maintain class distribution
- Test set locked for final evaluation only

### 2. Imbalance Handling Strategy

- `class_weight='balanced'` in Random Forest
- Stratified K-Fold Cross-Validation (k=5)
- Focus on both majority and minority class performance

### 3. Feature Selection

- **Method:** MDI (Mean Decrease in Impurity)
- Iterative testing from 1 to 17 features
- Selection based on custom priority order:
  1. Class 1 (Left) Recall
  2. Class 0 (Stayed) Recall
  3. Class 1 Precision
  4. Class 0 Precision
  5. Fewer Features
  6. F1 Scores
  7. Balanced Accuracy

### 4. Model Selection Criteria

- Performance thresholds: Recall ≥ 0.9 for both classes
- Optimal feature count for interpretability

---

## 📈 Results

### Selected Features (Top 5)

| Rank | Feature |
|------|---------|
| 1 | `satisfaction_level` |
| 2 | `time_spend_company` |
| 3 | `average_monthly_hours` |
| 4 | `number_project` |
| 5 | `last_evaluation` |

### Final Model Performance (Test Set)

| Metric | Class 0 (Stayed) | Class 1 (Left) |
|--------|------------------|----------------|
| **Recall** | 0.9980 | 0.9447 |
| **Precision** | 0.9891 | 0.9895 |
| **F1-Score** | 0.9935 | 0.9666 |

| Overall Metric | Score |
|----------------|-------|
| **Balanced Accuracy** | 0.9714 |
| **Overall Accuracy** | 0.9892 |

### Cross-Validation vs Test Performance

| Metric | CV Score | Test Score | Difference |
|--------|----------|------------|------------|
| Recall (Stayed) | 0.9981 | 0.9980 | -0.0001 |
| Recall (Left) | 0.9096 | 0.9447 | +0.0351 |
| Balanced Accuracy | 0.9539 | 0.9714 | +0.0175 |

✅ **Model generalizes well - no overfitting detected!**

---

## 💡 Key Findings

### Summary

The organization is facing a dual crisis of **burnout and stagnation**, systematically losing its most valuable employees. While low satisfaction (median 0.41 for leavers) is the immediate trigger, the root causes are structural: unsustainable workloads, a near-total lack of career progression, and compensation that fails to reward high effort.

### The Core Predictors of Turnover

| Metric | Employees Who Stayed | Employees Who Left | Key Insight |
|--------|----------------------|--------------------|-------------|
| Median Satisfaction | 0.69 | 0.41 | Low satisfaction is the common denominator for all departures |
| Median Evaluation | 0.71 | 0.79 | You are systematically losing your highest-performing employees |
| Median Monthly Hours | 198 | 226 | Leavers are pushed significantly harder |
| Promoted (Last 5 Yrs) | 1.8% | 0.3% | Promotions are a powerful yet neglected retention tool |

### The Workload Crisis: A U-Shaped Curve of Risk

| Risk Group | Hours/Month | Projects | Attrition Rate | Core Driver |
|------------|-------------|----------|----------------|-------------|
| 🔴 Extreme Burnout | 280–300+ | 6–7 | 62% – 100% | Unsustainable overload; 7 projects = 100% turnover |
| 🟢 The Safe Zone | 160–220 | 3–4 | 1% – 4% | Optimal balance with highest retention |
| 🟡 Under-Utilized | <160 | 2 | 30% – 54% | Boredom, disengagement, and poor role fit |

### Career Stagnation and the Tenure Cliff

| Tenure (Years) | Attrition Rate | Observation |
|----------------|----------------|-------------|
| 2 | 1.1% | High initial retention |
| 3 | 17.9% | First major spike in departures |
| 5 | 45.6% | Peak attrition point; nearly half of this cohort leaves |
| 6+ | 0.0% | Extreme loyalty among those who survive the cliff |

**The Promotion Paradox:** While only 1.69% of the workforce has been promoted, these employees have an attrition rate of 3.9%, compared to 16.8% for unpromoted staff.

### Compensation and Departmental Risks

| Salary Level | Attrition Rate | Workload Pattern | Key Insight |
|--------------|----------------|------------------|-------------|
| Low | 20.5% | More Hours: 223 vs. 197 | High performers are overworked and underpaid |
| Medium | 14.6% | More Hours: 229 vs. 199 | Moderate earners face similar burnout risks |
| High | 4.8% | Fewer Hours: 160 vs. 201 | Leavers are often under-utilized |

### Critical Flight Risk Profiles

| Cluster | Performance | Satisfaction | Workload | Primary Reason for Leaving |
|---------|-------------|--------------|----------|----------------------------|
| 🔥 The Burned Out | High (0.8–1.0) | Very Low (0.0–0.2) | Extreme Overtime | Exhaustion and lack of work-life balance |
| ⭐ The Poached Stars | High (0.8–1.0) | High (0.8–1.0) | High Overtime | Better external offers and lack of career growth |
| ❌ The Mismatched | Low (0.2–0.4) | Low (0.2–0.4) | Under-Utilized | Poor role fit or deep disengagement |

---

## 🎯 Strategic Recommendations

### 1. 🛑 Cap Workloads Immediately (Stop Burnout)

| Action | Details |
|--------|---------|
| **The Rule** | No employee should be assigned more than 4 projects or work more than 220 hours/month |
| **The Fix** | Flag anyone working >220 hours. Redistribute their work to under-utilized staff |
| **Why** | 7 projects or 300+ hours guarantees 100% turnover |

### 2. 📈 Fix the "Mid-Career" Promotion Gap

| Action | Details |
|--------|---------|
| **The Rule** | Implement a mandatory career review at the 3-year mark |
| **The Fix** | Create a clear promotion path. With only 1.69% of staff promoted in 5 years, experienced staff are forced to leave to advance |
| **Why** | Promoted employees have ~4% churn rate vs. ~17% for non-promoted |

### 3. 💵 Pay High Performers Fairly

| Action | Details |
|--------|---------|
| **The Rule** | Stop underpaying your hardest workers |
| **The Fix** | Audit Low and Medium salary brackets. Identify employees with High Evaluations (>0.8) and give them raises or bonuses |
| **Why** | You are losing top talent because they work the most hours but get the lowest pay |

### 4. 🎯 Engage the Under-Utilized

| Action | Details |
|--------|---------|
| **The Rule** | Identify employees with <160 hours or only 2 projects |
| **The Fix** | Assign them more work, upskill them, or manage them out |
| **Why** | Boredom is driving nearly as much turnover as burnout |

### 5. 🏢 Target High-Risk Departments

| Action | Details |
|--------|---------|
| **The Focus** | HR, Accounting, and Sales (highest churn) |
| **The Fix** | Conduct "Stay Interviews" in these specific departments to identify local stressors immediately |

---

<div align="center">
  <img src="https://github.com/JustToTryModels/IMAGES/blob/main/thank-you-33.gif?raw=true" alt="Thank You" width="400">
</div>
