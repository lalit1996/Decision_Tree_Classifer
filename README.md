🚗 Toyota Car Price Prediction using Machine Learning
📌 Project Overview

This project focuses on predicting Toyota car prices based on features such as model type, manufacturing year, and other attributes.
Multiple machine learning models were evaluated, with a strong emphasis on model validation and pruning techniques to improve generalization performance.

🎯 Objective

Predict car prices accurately

Compare Linear Regression and Decision Tree models

Reduce overfitting using Cost-Complexity Pruning

Select the best model using cross-validation

📊 Dataset Description

The dataset contains information about Toyota cars, including:

Model

Year

Other numerical and categorical features

Target variable: Car Price

🧠 Models Used
1️⃣ Linear Regression

Used as a baseline model

Achieved ~78% accuracy

Limited in capturing non-linear relationships

2️⃣ Decision Tree Regressor

Implemented with cost-complexity pruning

ccp_alpha values obtained using:

cost_complexity_pruning_path


Each ccp_alpha evaluated using 5-fold Stratified Cross-Validation

Final ccp_alpha selected based on average cross-validation score

✅ Achieved ~93% accuracy

🔬 Model Validation Strategy

Stratified K-Fold Cross Validation (5 folds)

Each pruning level (ccp_alpha) validated across all folds

Mean score used to ensure robust and unbiased performance

📈 Key Results
Model	Accuracy
Linear Regression	78%
Decision Tree (Optimized)	93%

✔ Significant performance improvement after pruning and validation
✔ Reduced overfitting
✔ Better generalization on unseen data

🛠️ Technologies & Libraries

Python

NumPy

Pandas

Scikit-learn

Matplotlib / Seaborn


🚀 Key Learnings

Proper hyperparameter tuning is critical

Cost-complexity pruning helps control tree overfitting

Cross-validation gives more reliable results than a single train-test split

Model simplicity often improves real-world performance

📌 Conclusion

This project demonstrates how systematic pruning and validation can significantly enhance model performance.
Decision Trees, when properly tuned, can outperform simpler models like Linear Regression for complex, non-linear datasets.
