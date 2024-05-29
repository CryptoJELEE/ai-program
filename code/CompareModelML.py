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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.model_selection import RandomizedSearchCV

def load_data():
    ecommerce_data = pd.read_csv('../data/rawdata_train.csv')
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
    X = data[['방문시간', '방문요일', '유입출처구분', '페이지뷰', '체류시간', '이동경로', '재방문여부']]
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
        'Decision Tree': RandomizedSearchCV(DecisionTreeClassifier(random_state=42), {
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }, n_iter=5, cv=3, n_jobs=-1, verbose=2, error_score='raise'),
        'Random Forest': RandomizedSearchCV(RandomForestClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }, n_iter=5, cv=3, n_jobs=-1, verbose=2, error_score='raise'),
        'k-NN': RandomizedSearchCV(KNeighborsClassifier(), {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance']
        }, n_iter=5, cv=3, n_jobs=-1, verbose=2, error_score='raise'),
        'Naive Bayes': GaussianNB(),
        'Neural Network': RandomizedSearchCV(MLPClassifier(random_state=42), {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }, n_iter=5, cv=3, n_jobs=-1, verbose=2, error_score='raise'),
        'XGBoost': RandomizedSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, tree_method='gpu_hist'), {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 0.9]
        }, n_iter=5, cv=3, n_jobs=-1, verbose=2, error_score='raise')
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

    return results, eval_times

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
    models, train_times = train_models(X_train, y_train)
    results, eval_times = evaluate_models(models, X_test, y_test)

    # Plot results
    plot_results(results, train_times, eval_times)

    # Display detailed results
    display_results(results)



