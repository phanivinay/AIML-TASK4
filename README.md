# AIML-TASK4
📌 Task 4: Classification with Logistic Regression
🎯 Objective

Build a binary classification model using Logistic Regression and evaluate its performance using standard ML metrics such as confusion matrix, precision, recall, F1-score, and ROC-AUC.
Additionally, understand threshold tuning and the sigmoid function used in logistic regression.

🛠️ Tools & Libraries Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

📂 Dataset

You can use any binary classification dataset.
For this task, we used the Breast Cancer Wisconsin Dataset, which is built into Scikit-learn.

🚀 Steps Performed
1️⃣ Choose a Binary Classification Dataset

Used the Breast Cancer dataset containing:

569 samples

30 input features

Target:

0: Malignant

1: Benign

2️⃣ Train/Test Split & Standardization

Split the dataset into 80% training and 20% testing.

Standardized features using StandardScaler to improve model performance.

3️⃣ Train Logistic Regression Model

Trained using:

LogisticRegression(max_iter=3000)


Logistic Regression learns a linear decision boundary using the sigmoid function.

4️⃣ Model Evaluation

Evaluated using:

✔ Confusion Matrix

✔ Precision

✔ Recall

✔ F1-score

✔ ROC Curve & AUC Score

These metrics help identify model accuracy and how well it distinguishes between classes.

5️⃣ Threshold Tuning

The default threshold = 0.5

Lowering threshold increases recall (good for medical predictions).

Demonstrated how predictions change with a custom threshold, e.g., 0.3.

6️⃣ Sigmoid Function Explanation

Logistic Regression uses the sigmoid function:

𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)=
1+e
−z
1
	​


This converts linear values into probabilities between 0 and 1.

📊 Output

Your output will include:

Confusion Matrix

Classification Report

ROC-AUC Score

ROC Curve Plot

Threshold-adjusted Confusion Matrix

Sigmoid example values
