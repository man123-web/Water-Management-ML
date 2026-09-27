# Water Management in Agriculture - ML Analysis
**JAIN UNIVERSITY | Department of IoT | USN: 23BTRCO059**

### Project Description
Machine Learning analysis of 1023 farmer survey responses from Karnataka to study water conservation practices. Dataset has 16 columns covering water source, irrigation technique, awareness and practice.

**Key Finding:** 79.6% aware but only 28.6% practice water conservation = 51% gap

### Dataset
- `WATER MANAGEMENT IN AGRICULTURE.csv`
- 1023 rows x 16 columns
- Target: Willingness to Adopt

### Model Results (Actual)
| Model | Accuracy | Macro-F1 |
|-------|----------|
| Logistic Regression | 0.6468 | 0.41 |
| Random Forest | 0.6219 | 0.37 |
| XGBoost | 0.6070 | 0.39 |

Note: IEEE 2024 paper 88.5% used IoT sensor data. Our survey data is behavioral so 64.68% is valid.

### How to Run
