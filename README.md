# E-Commerce User Action Prediction

**Authors**: Bang jung nam, Lee Seung jae 
**Date**: 2024/05/28

### Introduction
The purpose of this project is to classify the purchasing behavior of commerce users using deep learning algorithms, and to gain various insights by analyzing the characteristics of these classified users.

### prepare the Dataset 
Before training the model, We prepare data on e-commerce users' visit time, visit day, page views, session duration, traffic source, navigation path, and purchase status. Traffic sources are categorized numerically as bookmark, search ad, external domain, etc. Navigation paths are quantified by summing numbers assigned to the main page, detail page, and product list page.

### Compare Model List
We compare the accuracy and precision of each model by training the data using various machine learning algorithms such as logistic regression, decision tree, random forest, SVM, kNN, and XGBoost
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. SVM
5. kNN
6. XGBoost

## 1. Logistic Regression

#### Training Process:
```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)
```

### Testing
```python
# Make predictions on the test set
y_pred = model.predict(X_test)
```

### Individual Model Execute
```python
python3 LogisticReg.py
```

## Model Result
```result
훈련 데이터 크기 : (13279, 7)
테스트 데이터 크기 : (3320, 7)
Training Accuracy: 0.79
Test Accuracy: 0.80
Confusion Matrix: [[2056  141], [ 507  616]]
========================== 오차 행렬 ==============================
TN 2056 / FP 141
FN 507  / TP 616
정확도: 0.8048, 정밀도: 0.8137, 재현율: 0.5485, F1: 0.6553, AUC:0.8761
======================================================================
```
## Model Choice
The optimal prediction model was found to be the random forest.:
1. **Accuracy**: Calculate the accuracy of the logistic regression model.
2. **Prediction**: Predict if a person has diabetes or not based on their feature values.


## Data Details
- **Total Data**: 188,602 samples
- **Train Data**: Approximately 80% of the total data 
- **Test Data**: Remaining 20% of the total data
- **Other Test Data**: 5,886 samples


