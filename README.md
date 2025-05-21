# pass-fail-predictor

This project builds a machine learning pipeline to classify students into performance categories (`GradeClass`) based on various features such as study time, extracurricular activities, and parental background.

## Overview

* **Dataset**: `Student_performance.csv`
* **Goal**: Predict the `GradeClass` of students
* **Approach**:

  * Handle outliers and encode categorical variables
  * Use SMOTE for class balancing
  * Feature selection using Random Forest
  * Train a soft-voting ensemble model
  * Calibrate prediction probabilities
  * Tune class-wise thresholds for improved F1 score

## Models Used

* Logistic Regression (with Standard Scaling)
* Decision Tree
* Random Forest
* XGBoost

Combined using a weighted soft-voting ensemble classifier.

## Pipeline Steps

1. **Preprocessing**:

   * Cap outliers in `StudyTimeWeekly` and `Absences`
   * Frequency-encode categorical features
   * Drop original categorical columns

2. **Split**:

   * Train-test split (80-20) with stratification

3. **Balancing**:

   * Apply SMOTE to training data

4. **Feature Selection**:

   * Select features using Random Forest (median importance threshold)

5. **Training**:

   * Train individual classifiers
   * Build soft-voting ensemble

6. **Calibration**:

   * Apply isotonic calibration with 5-fold CV

7. **Threshold Tuning**:

   * Tune per-class thresholds using validation set

8. **Prediction**:

   * Use optimized thresholds for final predictions

9. **Evaluation**:

   * Accuracy, classification report, confusion matrix

10. **Visualization**:

    * Class distribution, feature importances, confusion matrix
    * Correlation analysis of selected features
    * Percentage distribution of categorical features by grade class

## Dependencies

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib shap
```

## Running the Code

Ensure the dataset `Student_performance.csv` is in the working directory. Then run the script:

```bash
python student_performance_classifier.py
```

## Output

* Printed accuracy and classification report
* Visual plots of:

  * Grade class distribution
  * Feature importances
  * Confusion matrix
  * Correlated feature relationships
  * Categorical feature breakdown by class

## Author

Developed by Varun Ekambaranath, Shivansh shah, adithya kommuri and tejas kollipara for student analytics and educational outcome prediction.
