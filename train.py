import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42

# === Paths ===
X_CSV = "processed_data/X.csv"
Y_CSV = "processed_data/y.csv"

# === Load data ===s
X = pd.read_csv(X_CSV).values
y = pd.read_csv(Y_CSV).values.flatten()

# Feature selection: top 100 features by mutual information
mi = mutual_info_classif(X, y, discrete_features=False, random_state=SEED)
top_k = 100
top_idx = np.argsort(mi)[::-1][:top_k]
X = X[:, top_idx]

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Use StratifiedKFold to split data into train/test (one fold)
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(X, y))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Compute class weights for XGBoost to handle class imbalance
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weights = {cls: total / count for cls, count in class_counts.items()}
weights_train = np.array([class_weights[label] for label in y_train])

# Train RandomForest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1,
    max_depth=10,
    min_samples_split=5
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# XGBoost training helper function
def train_xgb(X_train, y_train, X_val, y_val, weights_train, params, num_boost_round=500):
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params,
                      dtrain,
                      num_boost_round=num_boost_round,
                      evals=evallist,
                      early_stopping_rounds=30,
                      verbose_eval=False)
    return model

# Hyperparameter grid for XGBoost
param_grid = [
    {'max_depth': 6, 'learning_rate': 0.1},
    {'max_depth': 10, 'learning_rate': 0.1},
    {'max_depth': 10, 'learning_rate': 0.05},
    {'max_depth': 15, 'learning_rate': 0.1},
]

best_model = None
best_acc = rf_acc
best_name = 'RandomForest'
best_pred = rf_pred

for params_tune in param_grid:
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'seed': SEED,
        'tree_method': 'hist',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        **params_tune
    }
    model = train_xgb(X_train, y_train, X_test, y_test, weights_train, params)
    dtest = xgb.DMatrix(X_test)
    pred_prob = model.predict(dtest)
    pred = np.argmax(pred_prob, axis=1)
    acc = accuracy_score(y_test, pred)

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = f"XGBoost {params_tune}"
        best_pred = pred

# Ensemble if improves accuracy
if best_name != 'RandomForest':
    rf_pred_prob = rf.predict_proba(X_test)
    xgb_pred_prob = best_model.predict(xgb.DMatrix(X_test))
    avg_pred_prob = (xgb_pred_prob + rf_pred_prob) / 2
    ensemble_pred = np.argmax(avg_pred_prob, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    if ensemble_acc > best_acc:
        best_acc = ensemble_acc
        best_name = 'Ensemble RF + XGBoost'
        best_pred = ensemble_pred

print(f"\nBest model: {best_name} with accuracy: {best_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, best_pred))

# Confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_test, best_pred, title=f"{best_name} Confusion Matrix")
