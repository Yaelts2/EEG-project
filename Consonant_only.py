import numpy as np
import pandas as pd
from collections import Counter
import os

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42

# === Class name mapping ===
label_map = {
    0: 'a', 1: 'b', 2: 'd', 3: 'e', 4: 'i',
    5: 'o', 6: 'p', 7: 's', 8: 't', 9: 'u', 10: 'z'
}

# Define consonants
consonants = {'b','d','p','s','t','z'}

# === Paths ===
X_CSV = "processed_data/X.csv"
Y_CSV = "processed_data/y.csv"

# === Load data ===
X = pd.read_csv(X_CSV).values
y = pd.read_csv(Y_CSV).values.flatten()

# Map numeric labels to letters
letters = np.array([label_map[int(lbl)] for lbl in y])

# Filter only consonant samples
mask = np.isin(letters, list(consonants))
X = X[mask]
y = letters[mask]

print(f"Consonant classes in dataset: {set(y)}")
print(f"Total consonant samples: {len(y)}")

# === Remap consonants to 0..num_classes-1 for XGBoost ===
consonant_list = sorted(list(consonants))
letter_to_idx = {l:i for i,l in enumerate(consonant_list)}
idx_to_letter = {i:l for i,l in enumerate(consonant_list)}
y_num = np.array([letter_to_idx[l] for l in y])

# Feature selection: top 50 features by mutual information
mi = mutual_info_classif(X, y_num, discrete_features=False, random_state=SEED)
top_k = 50
top_idx = np.argsort(mi)[::-1][:top_k]
X = X[:, top_idx]

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified split: one fold for train/test
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)
train_idx, test_idx = next(skf.split(X, y_num))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_num[train_idx], y_num[test_idx]

# Compute class weights
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weights = {cls: total / count for cls, count in class_counts.items()}
weights_train = np.array([class_weights[label] for label in y_train])

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_split=5,
    random_state=SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# XGBoost training function
def train_xgb(X_train, y_train, X_val, y_val, weights_train, params, num_boost_round=200):
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params,
                      dtrain,
                      num_boost_round=num_boost_round,
                      evals=evals,
                      early_stopping_rounds=35,
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

for params in param_grid:
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': len(consonant_list),
        'eval_metric': 'mlogloss',
        'seed': SEED,
        'tree_method': 'hist',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        **params
    }
    model = train_xgb(X_train, y_train, X_test, y_test, weights_train, xgb_params)
    dtest = xgb.DMatrix(X_test)
    pred_prob = model.predict(dtest)
    pred = np.argmax(pred_prob, axis=1)
    acc = accuracy_score(y_test, pred)

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = f"XGBoost {params}"
        best_pred = pred

# Ensemble prediction if better
if best_name != 'RandomForest':
    rf_probs = rf.predict_proba(X_test)
    xgb_probs = best_model.predict(xgb.DMatrix(X_test))
    avg_probs = (rf_probs + xgb_probs) / 2
    ensemble_pred = np.argmax(avg_probs, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)

    if ensemble_acc > best_acc:
        best_acc = ensemble_acc
        best_name = 'Ensemble RF + XGBoost'
        best_pred = ensemble_pred

# === Print results with consonant letters ===
print(f"\nBest model: {best_name} with accuracy: {best_acc:.4f}")
target_names = consonant_list
report = classification_report([idx_to_letter[i] for i in y_test],
                               [idx_to_letter[i] for i in best_pred],
                               target_names=target_names)
print("\nClassification Report:")
print(report)

# Save results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, "Consonant_classification_report.txt"), "w") as f:
    f.write(f"Best model: {best_name}\n")
    f.write(f"Accuracy: {best_acc:.4f}\n\n")
    f.write(report)

# Plot and save confusion matrix with consonant names
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    labels = consonant_list
    cm = confusion_matrix([idx_to_letter[i] for i in y_true],
                          [idx_to_letter[i] for i in y_pred],
                          labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

plot_confusion_matrix(
    y_test,
    best_pred,
    title=f"{best_name} Confusion Matrix",
    save_path=os.path.join(results_dir, "Consonant_only_confusion_matrix.png")
)

print(f"\nResults saved in '{results_dir}' folder.")
