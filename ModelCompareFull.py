# -*- coding: utf-8 -*-
"""ai프로그래밍.ipynb의 사본

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oybQU8cFoelN9WUVqveT-t0-ug1l-QI6

# 단일 모델로 분석
  
- 모델 문제
  1. 단순한 모델
  2. 모델 지표 부족

- 데이터
  1. 데이터 불균형
  2. 변수 적절성

- 코드
  1. 코드 가독성
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

def get_clf_evel(y_test, pred=None, pred_proba = None):
  confusion = metrics.confusion_matrix(y_test, pred)
  accuracy = metrics.accuracy_score(y_test, pred)
  precision = metrics.precision_score(y_test,pred)
  recall = metrics.recall_score(y_test, pred)
  f1 = metrics.f1_score(y_test,pred)

  # ROC-AUC 추가
  roc_auc = metrics.roc_auc_score(y_test, pred_proba)
  print('오차 행렬')
  print(f'TN {confusion[0][0]}\t/ FP {confusion[0][1]}')
  print(f'FN {confusion[1][0]}\t/ TP {confusion[1][1]}')
  # ROC-AUC print 추가
  print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, \
          F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))

# Load the dataset
ecommerce_data = pd.read_csv('./data/rawdata_train.csv')
ecommerce_new = pd.read_csv('./data/rawdata_test.csv')

ecommerce_concat = pd.concat([ecommerce_data , ecommerce_new])
# 구매 자료와 비구매 자료를 별도로 추출
ecommerce_buy_y = ecommerce_concat[ecommerce_concat['구매여부']==1]
ecommerce_buy_nt = ecommerce_concat[ecommerce_concat['구매여부']==0]


# 비구매 자료는 잘라서 일부만 추출할거라,
# Shffle 처리. max_buy_count * N 배수 추출위해서.
ecommerce_buy_n = ecommerce_buy_nt.sample(frac=1)

# 구매 자료수 대비 *3 배수로 비구매자료를 추출해서 적용한다.
# 원래 자료는 3 : 97 (구매:비구매) 로 learn imbalanced (불균형) 발생함.
# Train 자료를 적합하도록 양자료간 비율 줄어야함.
max_buy_count = len(ecommerce_buy_y)*2
ecommerce_concat = pd.concat([ecommerce_buy_y , ecommerce_buy_n[:max_buy_count]])
ecommerce_member = ecommerce_concat

# 여러 자료 칼럼이 있을수 있어서 필요한 자료만 추출
# 세션구분키 : 각 방문자마다 구분되는 유니크키
# 방문시간 : 방문한 시간
# 유입출처구분 : 1 (bookmark), 2(검색광고), 3(무료검색), 4(배너광고), 5(외부 유입출처) 등.
# 페이지뷰 : 방문에 발생한 페이지뷰
# 체류시간 : 방문에 체류한시간  ( 2페이지 이상 본 + 마지막 페이지는 계산 안됨)
# 이동경로 : 컨텐츠 이동 경로 ( 1: 메인페이지(index 페이지등) , 2 : 리스트페이지, 3: 상품상세페이지 , 4: 장바구니,주문 등 )
# 재방문 여부 : 재방문 여부 ( 1: 재방문 , 0: 첫방문 )

X = ecommerce_member[['세션구분키', '방문시간', '이동경로','유입출처구분', '페이지뷰', '체류시간',  '재방문여부']]
y = ecommerce_member['구매여부']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Data Scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#sc = MinMaxScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

print("훈련 데이터 크기 :", X_train.shape)
print("테스트 데이터 크기 :", X_test.shape)
#print(X_train[:5])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)

# Display the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(y_pred[:500])

# Display results
print(f'Training Accuracy: {model.score(X_train, y_train):.2f}')
print(f'Test Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

y_pred_proba = model.predict_proba(X_test)[::,1]
print(y_pred_proba[:500])
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr, label = "AUC="+str(auc))
plt.legend(loc=4)
plt.show()

get_clf_evel(y_test,y_pred, y_pred_proba)


# 구매여부가 파악이 안된 목록
ecommerce_new = pd.read_csv('./data/rawdata_test.csv')

# 구매 자료와 비구매 자료를 별도로 추출
ecommerce_nbuy_y = ecommerce_new[ecommerce_new['구매여부']==1]
ecommerce_nbuy_n = ecommerce_new[ecommerce_new['구매여부']==0]

# 구매 자료수 대비 *3 배수로 비구매자료를 추출해서 적용한다.
# 원래 자료는 3 : 97 (구매:비구매) 로 learn imbalanced (불균형) 발생함.
# Train 자료를 적합하도록 양자료간 비율 줄어야함.
max_buy_count = len(ecommerce_nbuy_y)*3
ecommerce_n_concat = pd.concat([ecommerce_nbuy_y , ecommerce_nbuy_n[:max_buy_count]])

X2_Data = ecommerce_new[ecommerce_new['방문시간']==12]
#X2_Data = ecommerce_n_concat
#X2_Data = ecommerce_new[ecommerce_new['구매여부']==1]
#X2_Data = ecommerce_new
X_New = X2_Data[['세션구분키', '방문시간','이동경로', '유입출처구분', '페이지뷰', '체류시간',  '재방문여부']]
y_New = X2_Data['구매여부']
#new_data = X_New[:1000]

#print(new_data)
X_New = scaler.transform(X_New)
#X_New = sc.transform(X_New)

# Make predictions on the new data
new_data_predictions = model.predict(X_New)
accuracyNew = metrics.accuracy_score(y_New, new_data_predictions)
conf_matrixNew = metrics.confusion_matrix(y_New, new_data_predictions)

print(f'TestNew Accuracy: {accuracyNew:.2f}')
print('Confusion Matrix:')
print(conf_matrixNew)

y2_pred_proba = model.predict_proba(X_New)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_New, y2_pred_proba)
auc = metrics.roc_auc_score(y_New, y2_pred_proba)

plt.plot(fpr,tpr, label = "AUC="+str(auc))
plt.legend(loc=4)
plt.show()

# Display predictions for the new data
new_data_with_predictions = X_New.copy()
#new_data_with_predictions['구매여부'] = new_data_predictions
#print('Predictions for New Data:')
#print(new_data_with_predictions[['세션구분키', '구매여부']])

"""# 모델 개선
  
- 모델 문제
  1. 모델 단순성 -> 다양한 분류 모델 사용
  2. 모델 지표 부족 -> 지표 추가


- 데이터
  1. 데이터 불균형 -> "smote" 함수 사용
  2. 변수 적절성

- 코드
  1. 코드 가독성 -> 코드 분리

모델 호출
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.model_selection import RandomizedSearchCV

"""데이터 로드"""

def load_data():
    ecommerce_data = pd.read_csv('./data/rawdata_train.csv')

    return ecommerce_data

print(ecommerce_data)
print(ecommerce_new)

"""전처리"""

def preprocess_data(ecommerce_data):
    ecommerce_buy_y = ecommerce_data[ecommerce_data['구매여부'] == 1]
    ecommerce_buy_nt = ecommerce_data[ecommerce_data['구매여부'] == 0]
    max_buy_count = len(ecommerce_buy_y)
    ecommerce_buy_n = ecommerce_buy_nt.sample(n=max_buy_count*2, random_state=42)
    ecommerce_buy = pd.concat([ecommerce_buy_y, ecommerce_buy_n])
    ecommerce_member = ecommerce_buy.sample(frac=1, random_state=42)  # Shuffle
    return ecommerce_member

"""피처 설정"""

def feature_selection(data):
    X = data[['유입출처구분', '페이지뷰', '체류시간', '이동경로', '재방문여부']]
    y = data['구매여부']
    return X, y

"""데이터 분리"""

def train_test_splitting(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

"""데이터 불균형 해소

---
SMOTE의 동작 방식은 데이터의 개수가 적은 클래스의 표본을 가져온 뒤 임의의 값을 추가하여 새로운 샘플을 만들어 데이터에 추가하는 오버샘플링 방식이다.

https://john-analyst.medium.com/smote%EB%A1%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%88%EA%B7%A0%ED%98%95-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0-5ab674ef0b32

"""

def handle_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

"""스케일링하기

훈련, 테스트테이터 세트 스케일링
"""

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler

"""로지스틱 회귀 (Logistic Regression) 설명: 로지스틱 회귀는 이진 분류 문제에서 많이 사용되는 선형 모델입니다. 입력된 특징들의 가중합을 로지스틱 함수에 통과시켜 두 클래스 중 하나에 속할 확률을 예측합니다.
사용 경우: 이진 분류 문제, 확률 예측이 필요한 경우.

의사결정 나무 (Decision Tree) 설명: 의사결정 나무는 데이터의 특징들에 따라 데이터를 분할하여 예측을 수행하는 트리 구조의 모델입니다. 트리의 각 노드는 특정 특징에 대한 분기 기준을 나타냅니다.
사용 경우: 데이터의 특징에 따라 명확한 분류 기준이 있는 경우.

랜덤 포레스트 (Random Forest) 설명: 랜덤 포레스트는 여러 개의 의사결정 나무를 앙상블하여 예측의 정확성을 높이는 모델입니다. 각 나무는 무작위로 선택된 데이터와 특징을 사용하여 학습됩니다.
사용 경우: 과적합을 줄이고 예측의 정확성을 높이고 싶은 경우.

SVM (서포트 벡터 머신, Support Vector Machine) 설명: SVM은 데이터를 고차원 공간으로 매핑하여 두 클래스 간의 최대 마진을 찾는 분류 모델입니다. 비선형 분류를 위해 커널 트릭을 사용할 수 있습니다. 사용 경우: 고차원 데이터, 비선형 분류 문제.

k-NN (k-최근접 이웃, k-Nearest Neighbors) 설명: k-NN은 새로운 데이터 포인트를 예측할 때, 가장 가까운 k개의 이웃 데이터를 참고하여 다수결로 분류를 수행하는 비모수 모델입니다. 사용 경우: 단순한 분류 문제, 데이터의 분포를 파악하고 싶은 경우.

나이브 베이즈 (Naive Bayes) 설명: 나이브 베이즈는 특징들이 서로 독립이라는 가정 하에 베이즈 정리를 적용하여 분류를 수행하는 모델입니다. 주로 텍스트 분류에 많이 사용됩니다. 사용 경우: 텍스트 분류, 스팸 필터링.

신경망 (Neural Network) 설명: 신경망은 입력층, 은닉층, 출력층으로 구성된 노드들의 네트워크입니다. 각 노드는 가중치를 가지며 활성화 함수를 통해 신호를 전달합니다. 복잡한 비선형 관계를 학습할 수 있습니다. 사용 경우: 이미지 인식, 자연어 처리 등 복잡한 패턴 인식 문제.

XGBoost (Extreme Gradient Boosting) 설명: XGBoost는 그래디언트 부스팅 알고리즘을 기반으로 한 강력한 앙상블 모델입니다. 여러 약한 학습기(주로 의사결정 나무)를 순차적으로 학습시키며, 각 단계에서 이전 단계의 오차를 줄이는 방향으로 모델을 개선합니다. 사용 경우: 대규모 데이터셋, 높은 예측 성능이 필요한 경우.
"""

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': RandomizedSearchCV(DecisionTreeClassifier(random_state=42), {
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }, n_iter=10, cv=3, n_jobs=-1, verbose=2, error_score='raise'),
        'Random Forest': RandomizedSearchCV(RandomForestClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }, n_iter=10, cv=3, n_jobs=-1, verbose=2, error_score='raise'),
        'k-NN': RandomizedSearchCV(KNeighborsClassifier(), {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance']
        }, n_iter=10, cv=3, n_jobs=-1, verbose=2, error_score='raise'),
        'Naive Bayes': GaussianNB(),
        'Neural Network': RandomizedSearchCV(MLPClassifier(random_state=42), {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }, n_iter=10, cv=3, n_jobs=-1, verbose=2, error_score='raise'),
        'XGBoost': RandomizedSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, tree_method='hist'), {  # CPU 모드 사용
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 0.9]
        }, n_iter=10, cv=3, n_jobs=-1, verbose=2, error_score='raise')
    }

    best_models = {}
    train_times = {}
    for name, model in models.items():
        print(f'Training {name}...')
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            end_time = time.time()
            train_times[name] = end_time - start_time
            if hasattr(model, 'best_estimator_'):
                best_models[name] = model.best_estimator_
            else:
                best_models[name] = model
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    return best_models, train_times

"""평가하기"""

def evaluate_models(models, X_test, y_test):
    results = {}
    eval_times = {}
    for name, model in models.items():
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        end_time = time.time()
        eval_times[name] = end_time - start_time
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        results[name] = {
            'accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'class_report': class_report,
            'roc_auc': roc_auc
        }

    return results, eval_times

"""시각화"""

def plot_results(results, train_times, eval_times):
    accuracies = [res['accuracy'] for res in results.values()]
    roc_aucs = [res['roc_auc'] for res in results.values() if res['roc_auc'] is not None]
    names = list(results.keys())

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    sns.barplot(x=accuracies, y=names)
    plt.title('Model Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Model')

    plt.subplot(2, 2, 2)
    sns.barplot(x=roc_aucs, y=names[:len(roc_aucs)])
    plt.title('Model ROC-AUC')
    plt.xlabel('ROC-AUC')
    plt.ylabel('Model')

    train_times_sorted = sorted(train_times.items(), key=lambda item: item[1], reverse=True)
    eval_times_sorted = sorted(eval_times.items(), key=lambda item: item[1], reverse=True)
    train_names, train_values = zip(*train_times_sorted)
    eval_names, eval_values = zip(*eval_times_sorted)

    plt.subplot(2, 2, 3)
    sns.barplot(x=train_values, y=train_names)
    plt.title('Model Training Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Model')

    plt.subplot(2, 2, 4)
    sns.barplot(x=eval_values, y=eval_names)
    plt.title('Model Evaluation Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Model')

    plt.tight_layout()
    plt.show()

def display_results(results):
    for model, metrics in results.items():
        print(f"Model: {model}")
        print(f"Accuracy: {metrics['accuracy']}")
        print("Confusion Matrix:")
        print(metrics['conf_matrix'])
        print("Classification Report:")
        print(pd.DataFrame(metrics['class_report']).transpose())
        if metrics['roc_auc'] is not None:
            print(f"ROC-AUC: {metrics['roc_auc']}")
        print("\n")

def display_feature_importance(models, feature_names):
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame(importances, index=feature_names, columns=['Importance']).sort_values(by='Importance', ascending=False)
            print(f'Feature importances for {model_name}:\n{importance_df}\n')
        elif hasattr(model, 'coef_'):
            importances = model.coef_[0]
            importance_df = pd.DataFrame(importances, index=feature_names, columns=['Importance']).sort_values(by='Importance', ascending=False)
            print(f'Feature importances for {model_name}:\n{importance_df}\n')

if __name__ == "__main__":
    # Load and preprocess data
    ecommerce_data = load_data()
    ecommerce_member = preprocess_data(ecommerce_data)

    # Feature selection and data splitting
    X, y = feature_selection(ecommerce_member)
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_splitting(X, y)

    # Handle class imbalance and scale features
    X_train, y_train = handle_imbalance(X_train, y_train)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Train and evaluate models
    models, train_times = train_models(X_train, y_train)
    results, eval_times = evaluate_models(models, X_test, y_test)

    # Plot results
    plot_results(results, train_times, eval_times)

    # Display detailed results
    display_results(results)

    # Display feature importance
    display_feature_importance(models, feature_names)


