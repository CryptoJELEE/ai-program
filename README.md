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

## Compare Model Result

```result
Mode Logistic Regression : 정확도 = 0.8067617866004962,정밀도 = 0.7243035542747358,재현율 = 0.691743119266055,F1 스코어 = 0.7076489910839981,  
Mode Decision Tree : 정확도 = 0.7921836228287841,정밀도 = 0.6548672566371682,재현율 = 0.8146788990825689,F1 스코어 = 0.7260834014717906,  
Mode Random Forest : 정확도 = 0.8052109181141439,정밀도 = 0.6824644549763034,재현율 = 0.7926605504587156,F1 스코어 = 0.733446519524618,  
Mode k-NN : 정확도 = 0.7636476426799007,정밀도 = 0.6277258566978193,재현율 = 0.7394495412844037,F1 스코어 = 0.6790227464195451,  
Mode Naive Bayes : 정확도 = 0.7872208436724566,정밀도 = 0.7556962025316456,재현율 = 0.5477064220183486,F1 스코어 = 0.6351063829787233,  
Mode Neural Network : 정확도 = 0.8061414392059554,정밀도 = 0.6744186046511628,재현율 = 0.8247706422018348,F1 스코어 = 0.7420553033429633,  
Model: Logistic Regression
Accuracy: 0.8067617866004962
Confusion Matrix:
[[1847  287]
 [ 336  754]]
Classification Report:
              precision    recall  f1-score      support
0              0.846083  0.865511  0.855687  2134.000000
1              0.724304  0.691743  0.707649  1090.000000
accuracy       0.806762  0.806762  0.806762     0.806762
macro avg      0.785193  0.778627  0.781668  3224.000000
weighted avg   0.804911  0.806762  0.805637  3224.000000
ROC-AUC: 0.8777069379121778


Model: Decision Tree
Accuracy: 0.7921836228287841
Confusion Matrix:
[[1666  468]
 [ 202  888]]
Classification Report:
              precision    recall  f1-score      support
0              0.891863  0.780694  0.832584  2134.000000
1              0.654867  0.814679  0.726083  1090.000000
accuracy       0.792184  0.792184  0.792184     0.792184
macro avg      0.773365  0.797686  0.779334  3224.000000
weighted avg   0.811737  0.792184  0.796577  3224.000000
ROC-AUC: 0.853914989295203


Model: Random Forest
Accuracy: 0.8052109181141439
Confusion Matrix:
[[1732  402]
 [ 226  864]]
Classification Report:
              precision    recall  f1-score      support
0              0.884576  0.811621  0.846530  2134.000000
1              0.682464  0.792661  0.733447  1090.000000
accuracy       0.805211  0.805211  0.805211     0.805211
macro avg      0.783520  0.802141  0.789988  3224.000000
weighted avg   0.816244  0.805211  0.808298  3224.000000
ROC-AUC: 0.8833205076395279


Model: k-NN
Accuracy: 0.7636476426799007
Confusion Matrix:
[[1656  478]
 [ 284  806]]
Classification Report:
              precision    recall  f1-score      support
0              0.853608  0.776007  0.812960  2134.000000
1              0.627726  0.739450  0.679023  1090.000000
accuracy       0.763648  0.763648  0.763648     0.763648
macro avg      0.740667  0.757729  0.745991  3224.000000
weighted avg   0.777240  0.763648  0.767677  3224.000000
ROC-AUC: 0.8310239202772071


Model: Naive Bayes
Accuracy: 0.7872208436724566
Confusion Matrix:
[[1941  193]
 [ 493  597]]
Classification Report:
              precision    recall  f1-score      support
0              0.797453  0.909560  0.849825  2134.000000
1              0.755696  0.547706  0.635106  1090.000000
accuracy       0.787221  0.787221  0.787221     0.787221
macro avg      0.776574  0.728633  0.742466  3224.000000
weighted avg   0.783335  0.787221  0.777231  3224.000000
ROC-AUC: 0.8033550295349217


Model: Neural Network
Accuracy: 0.8061414392059554
Confusion Matrix:
[[1700  434]
 [ 191  899]]
Classification Report:
              precision    recall  f1-score      support
0              0.898995  0.796626  0.844720  2134.000000
1              0.674419  0.824771  0.742055  1090.000000
accuracy       0.806141  0.806141  0.806141     0.806141
macro avg      0.786707  0.810698  0.793388  3224.000000
weighted avg   0.823068  0.806141  0.810010  3224.000000
ROC-AUC: 0.8915745939485654

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

## NOTE
- We used the SMOTE method to balance the data.

본 연구에서는 다양한 머신러닝 모델을 적용하여 주어진 데이터셋의 예측 성능을 평가하였다. 각 모델의 성능은 정확도(Accuracy), ROC-AUC, 혼동 행렬(Confusion Matrix), 분류 보고서(Classification Report)를 통해 측정하였다. 아래는 각 모델별 성능 평가 결과 및 활용 방안이다.

## 1. 로지스틱 회귀 (Logistic Regression)
정확도: 79.2%
ROC-AUC: 87.6%
특징: 해석이 용이하고 빠른 훈련 속도를 가지며, 준수한 정확도와 ROC-AUC 점수를 보인다.
활용 방안:
모델의 해석 가능성이 중요할 때 사용. 예를 들어, 특정 피처가 결과에 미치는 영향을 설명할 필요가 있을 때 적합하다.
비교적 간단한 문제를 해결할 때 유용하다.

## 2. 의사결정나무 (Decision Tree)
정확도: 77.3%
ROC-AUC: 81.9%
특징: 직관적으로 이해하기 쉽고, 시각화가 가능하나 복잡한 데이터에서는 과적합(overfitting) 가능성이 있다.
활용 방안:
데이터의 피처 중요도를 설명할 때 유용하다.
간단한 의사결정을 시각적으로 보여줄 필요가 있을 때 적합하다.

## 3. 랜덤 포레스트 (Random Forest)
정확도: 80.3%
ROC-AUC: 88.5%
특징: 강력한 성능과 과적합 방지 능력을 가지며, 다양한 데이터에 대해 안정적인 성능을 보인다.
활용 방안:
다양한 피처를 가진 복잡한 데이터셋을 처리할 때 적합하다.
피처 중요도를 제공하여 중요한 변수를 식별하는 데 유용하다.

## 4. k-최근접 이웃 (k-NN)
정확도: 76.5%
ROC-AUC: 82.6%
특징: 이해하기 쉽고 비선형 분포도 잘 처리하지만, 대용량 데이터에서 계산 비용이 높다.
활용 방안:
비선형 데이터 분포를 처리할 때 유용하다.
메모리가 충분한 경우 및 실시간 예측이 필요하지 않을 때 적합하다.

## 5. 나이브 베이즈 (Naive Bayes)
정확도: 78.2%
ROC-AUC: 80.5%
특징: 훈련 속도가 빠르고 비교적 간단한 모델로, 독립 가정이 강하지만 많은 경우에 잘 작동한다.
활용 방안:
텍스트 분류와 같은 높은 차원의 데이터에서 유용하다.
속도가 중요한 경우 적합하다.

## 6. 신경망 (Neural Network)
정확도: 80.8%
ROC-AUC: 88.9%
특징: 복잡한 패턴을 잘 학습하지만, 훈련에 시간이 걸리고 많은 데이터가 필요하다.
활용 방안:
복잡한 데이터 패턴을 학습해야 할 때, 예를 들어 이미지나 자연어 처리에 유용하다.
성능이 중요한 경우 및 충분한 데이터와 계산 자원이 있을 때 적합하다.

## 종합 제안
본 연구에서는 다양한 모델의 성능을 비교 분석한 결과, 랜덤 포레스트와 신경망 모델이 가장 높은 성능을 보였다. 따라서 이 두 모델을 주요 모델로 선택하여 활용하는 것이 바람직하다. 또한, 해석 가능성과 성능의 균형이 중요할 경우 로지스틱 회귀나 랜덤 포레스트를, 단순한 구현과 빠른 예측이 필요할 경우 나이브 베이즈를 고려할 수 있다.랜덤포레스트, 신경망 예측모델 기반 앙상블 결과 유의미한 차이 없음.


## Future Direction Using Model

We plan to handle modules for each model as classes and run them as daemons to continuously train. 
Through a separate web service interface, visitor data will be transmitted to extract purchase predictions and probabilities. 
This information will be used to select targets for post-event analysis and for separate marketing initiatives.
- Class Modeling and Deamon Program
- Interface Design
- Integration With WebService 

## MLP E-Commerce User Purchase Prediction Diagram

```mermaid
sequenceDiagram
    participant ModelDeamon
    participant JsonInterface
    participant WebService
    WebService->>JsonInterface: Request Predict
    JsonInterface->>ModelDeamon: Request Predict
    ModelDeamon->>JsonInterface: Response Predict Result
    JsonInterface->>WebService: Response Predict Result    
```

