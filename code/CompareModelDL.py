import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def load_data():
    # Load the dataset
    # Train / Test Dataset
    ecommerce_data = pd.read_csv('../data/rawdata_train.csv')
    # other Test Dataset
    ecommerce_new = pd.read_csv('../data/rawdata_test.csv')
    return ecommerce_data, ecommerce_new

def preprocess_data(ecommerce_data):
    ecommerce_buy_y = ecommerce_data[ecommerce_data['구매여부'] == 1]
    ecommerce_buy_nt = ecommerce_data[ecommerce_data['구매여부'] == 0]
    max_buy_count = len(ecommerce_buy_y)
    ecommerce_buy_n = ecommerce_buy_nt.sample(n=max_buy_count*2, random_state=42)
    ecommerce_buy = pd.concat([ecommerce_buy_y, ecommerce_buy_n])
    ecommerce_member = ecommerce_buy.sample(frac=1, random_state=42)  # Shuffle
    return ecommerce_member

def feature_selection(data):
    X = data[['방문시간', '유입출처구분', '페이지뷰', '체류시간', '이동경로', '재방문여부']]
    y = data['구매여부']
    return X, y

def train_test_splitting(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def handle_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': GridSearchCV(DecisionTreeClassifier(random_state=42), {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }, cv=5, n_jobs=-1, verbose=2),
        'Random Forest': GridSearchCV(RandomForestClassifier(random_state=42), {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }, cv=5, n_jobs=-1, verbose=2),
        #'SVM': GridSearchCV(SVC(probability=True, random_state=42), {
        #    'C': [0.1, 1, 10, 100],
        #    'gamma': [1, 0.1, 0.01, 0.001],
        #    'kernel': ['rbf', 'linear']
        #}, cv=5, n_jobs=-1, verbose=2),
        'k-NN': GridSearchCV(KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }, cv=5, n_jobs=-1, verbose=2),
        'Naive Bayes': GaussianNB(),
        'Neural Network': GridSearchCV(MLPClassifier(random_state=42), {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }, cv=5, n_jobs=-1, verbose=2),
        'XGBoost': GridSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False), {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }, cv=5, n_jobs=-1, verbose=2)
    }

    best_models = {}
    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X_train, y_train)
        if hasattr(model, 'best_estimator_'):
            best_models[name] = model.best_estimator_
            print("="*30)
            print(f'{name} 파라미터: {model.best_params_}')
            print(name, '예측 정확도: {:.4f}'.format(model.best_score_))
        else:
            best_models[name] = model


    return best_models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        F1 = f1_score(y_test, y_pred)
        print(f"Mode {name} : 정확도 = {accuracy},정밀도 = {precision},재현율 = {recall},F1 스코어 = {F1},  ")

        results[name] = {
            'accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'class_report': class_report,
            'roc_auc': roc_auc
        }

    return results

def plot_results(results):
    accuracies = [res['accuracy'] for res in results.values()]
    roc_aucs = [res['roc_auc'] for res in results.values() if res['roc_auc'] is not None]
    names = list(results.keys())

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(x=accuracies, y=names)
    plt.title('Model Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Model')

    plt.subplot(1, 2, 2)
    sns.barplot(x=roc_aucs, y=names[:len(roc_aucs)])
    plt.title('Model ROC-AUC')
    plt.xlabel('ROC-AUC')
    plt.ylabel('Model')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    ecommerce_data, ecommerce_new = load_data()
    ecommerce_member = preprocess_data(ecommerce_data)

    # Feature selection and data splitting
    X, y = feature_selection(ecommerce_member)
    X_train, X_test, y_train, y_test = train_test_splitting(X, y)

    # Handle class imbalance and scale features
    X_train, y_train = handle_imbalance(X_train, y_train)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Train and evaluate models
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    # Plot results
    plot_results(results)

