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
  print('========================== 오차 행렬 ==============================')
  print(f'TN {confusion[0][0]}\t/ FP {confusion[0][1]}')
  print(f'FN {confusion[1][0]}\t/ TP {confusion[1][1]}')
  # ROC-AUC print 추가
  print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
  print("="*70)

# Load the dataset
ecommerce_data = pd.read_csv('../data/rawdata_train.csv')
ecommerce_new = pd.read_csv('../data/rawdata_test.csv')

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
# 방문시간 : 방문한 시간 ( 시간 만 0~ 23 시 )
# 방문요일 : 방문한 요일 ( 1:월요일, 2:화요일.... 7:일요일 )
# 유입출처구분 : 1 (bookmark), 2(검색광고), 3(무료검색), 4(배너광고), 5(외부 유입출처) 등.
# 페이지뷰 : 방문에 발생한 페이지뷰
# 체류시간 : 방문에 체류한시간  ( 2페이지 이상 본 + 마지막 페이지는 계산 안됨)
# 이동경로 : 이동한 페이지 조합 값( 1: 메인페이지(index 페이지등) , 2 : 리스트페이지, 4: 상품상세페이지 등 ) ,예) 1 또는 2 또는 4 모두 이동했을때 7 값으로 추출
# 재방문 여부 : 재방문 여부 ( 1: 재방문 , 0: 첫방문 )

X = ecommerce_member[['세션구분키', '방문시간','방문요일', '유입출처구분', '페이지뷰', '체류시간',  '재방문여부']]
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
#print(y_pred[:500])

# Display results
print(f'Training Accuracy: {model.score(X_train, y_train):.2f}')
print(f'Test Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

'''
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
'''

y_pred_proba = model.predict_proba(X_test)[::,1]
#print(y_pred[:500])
#print(y_pred_proba[:500])
'''
for i in range(len(y_pred)):
    if y_pred[i] == 1 :
        print(f'{i} : {y_pred[i]} : {y_pred_proba[i]:.2f} : {y_test.values[i]}')
'''

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

'''
plt.plot(fpr,tpr, label = "AUC="+str(auc))
plt.legend(loc=4)
plt.show()
'''

get_clf_evel(y_test,y_pred, y_pred_proba)

X2_Data = ecommerce_new
X_New = X2_Data[['세션구분키', '방문시간','방문요일', '유입출처구분', '페이지뷰', '체류시간',  '재방문여부']]
y_New = X2_Data['구매여부']

X_New = scaler.transform(X_New)

# Make predictions on the new data
y2_pred = model.predict(X_New)
accuracyNew = metrics.accuracy_score(y_New, y2_pred )
conf_matrixNew = metrics.confusion_matrix(y_New, y2_pred )

print("="*70)
print(f'NextDay Dataset Accuracy: {accuracyNew:.2f}')
print('Confusion Matrix:')
print(conf_matrixNew)

y2_pred_proba = model.predict_proba(X_New)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_New, y2_pred_proba)
auc = metrics.roc_auc_score(y_New, y2_pred_proba)

'''
plt.plot(fpr,tpr, label = "AUC="+str(auc))
plt.legend(loc=4)
plt.show()
'''

get_clf_evel(y_New,y2_pred, y2_pred_proba)

