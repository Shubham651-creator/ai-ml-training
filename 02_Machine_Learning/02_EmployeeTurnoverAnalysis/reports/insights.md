# Employee Turnover Analytics - Final Report

## Project Objective

The objective of this project was to analyze employee attrition patterns and build a Machine Learning model capable of predicting whether an employee is likely to leave the organization. The analysis was performed using the IBM HR Analytics Employee Attrition dataset.

---

# Dataset Overview

* Total Employees: 1470
* Problem Type: Binary Classification
* Target Variable: Attrition

  * 1 = Employee Leaves
  * 0 = Employee Stays

---

# Exploratory Data Analysis (EDA) Findings

## 1. Overtime Significantly Impacts Attrition

Employees working overtime showed a much higher probability of leaving the company.

| Overtime Status | Attrition Rate |
| --------------- | -------------- |
| No Overtime     | 10.43%         |
| Overtime        | 30.53%         |

### Insight

Employees performing overtime are approximately three times more likely to leave compared to employees with normal workloads.

---

## 2. Salary Strongly Influences Retention

Attrition decreases as employee income increases.

| Monthly Income Range | Attrition Rate |
| -------------------- | -------------- |
| Lowest Income Group  | 31.29%         |
| Highest Income Group | 9.18%          |

### Insight

Lower-income employees are considerably more likely to resign than higher-income employees.

---

## 3. Business Travel Increases Attrition Risk

| Travel Frequency  | Attrition Rate |
| ----------------- | -------------- |
| Non-Travel        | 8.00%          |
| Travel Rarely     | 14.96%         |
| Travel Frequently | 24.91%         |

### Insight

Employees who travel frequently are more likely to leave the organization.

---

## 4. Job Satisfaction Affects Employee Retention

| Job Satisfaction Level | Attrition Rate |
| ---------------------- | -------------- |
| Level 1                | 22.84%         |
| Level 4                | 11.33%         |

### Insight

Higher job satisfaction is associated with lower employee turnover.

---

## 5. Employee Tenure Matters

Average years spent at company:

| Employee Status | Average Years |
| --------------- | ------------- |
| Stayed          | 7.37 Years    |
| Left            | 5.13 Years    |

### Insight

Employees who leave tend to have shorter tenure within the organization.

---

## 6. Department Analysis

| Department             | Attrition Rate |
| ---------------------- | -------------- |
| Sales                  | 20.63%         |
| Human Resources        | 19.05%         |
| Research & Development | 13.84%         |

### Insight

The Sales department experiences the highest attrition rate among all departments.

---

# Machine Learning Models Evaluated

## Logistic Regression

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 86.05% |
| Precision | 61.54% |
| Recall    | 34.04% |
| F1 Score  | 43.84% |

---

## Logistic Regression (Balanced)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 75.17% |
| Precision | 34.52% |
| Recall    | 61.70% |
| F1 Score  | 44.27% |

### Observation

This model achieved the highest Recall and successfully identified more employees likely to leave.

---

## Decision Tree

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 83.33% |
| Precision | 44.44% |
| Recall    | 17.02% |
| F1 Score  | 24.62% |

---

## Decision Tree (Balanced)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 77.21% |
| Precision | 35.71% |
| Recall    | 53.19% |
| F1 Score  | 42.74% |

---

## Random Forest

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 82.31% |
| Precision | 45.10% |
| Recall    | 48.94% |
| F1 Score  | 46.94% |

---

# Model Selection

Although Random Forest achieved the highest F1 Score, the primary business objective of this project is to identify employees who are likely to leave.

Therefore, Recall is considered more important than Accuracy.

The Logistic Regression model with class balancing achieved the highest Recall score:

* Recall = 61.70%

This means the model successfully identifies a larger proportion of employees who are at risk of leaving the company.

### Selected Model

**Logistic Regression (class_weight='balanced')**

---

# Feature Importance Analysis

The Random Forest model identified the following features as most influential:

1. MonthlyIncome
2. Age
3. TotalWorkingYears
4. DailyRate
5. YearsAtCompany
6. MonthlyRate
7. HourlyRate
8. DistanceFromHome
9. YearsWithCurrManager
10. OverTime

### Insight

Employee attrition is primarily influenced by:

* Compensation
* Experience level
* Company tenure
* Workload
* Distance from workplace

---

# Business Recommendations

Based on the analysis, the company should consider:

1. Reducing excessive overtime requirements.
2. Improving compensation for lower-income employees.
3. Monitoring employees with frequent business travel.
4. Increasing employee engagement and job satisfaction initiatives.
5. Focusing retention programs on newer employees.
6. Investigating attrition drivers within the Sales department.

---

# Conclusion

This project successfully applied the complete Machine Learning workflow:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Data Preprocessing
* Classification Modeling
* Model Evaluation
* Feature Importance Analysis

The analysis revealed that compensation, overtime, business travel, job satisfaction, and employee tenure are the most significant factors contributing to employee attrition.

A balanced Logistic Regression model was selected as the final model because it best aligns with the business goal of identifying employees at risk of leaving and enabling proactive retention strategies.
