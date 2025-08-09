import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# === Paths ===
X_CSV = "processed_data/X.csv"
Y_CSV = "processed_data/y.csv"

# === Load data ===
X = pd.read_csv(X_CSV).values
y = pd.read_csv(Y_CSV).values.flatten()

print(f"✅ Loaded data shape: X={X.shape}, y={y.shape}")

# === Feature scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"✅ Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# === Cross-validation setup ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === RandomForest classifier and hyperparameter grid ===
rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("\n🔎 Running GridSearchCV for RandomForest...")
grid_rf = GridSearchCV(rf_clf, param_grid_rf, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)
print(f"✅ Best RF params: {grid_rf.best_params_}")
print(f"✅ Best RF CV accuracy: {grid_rf.best_score_:.4f}")

# === Evaluate RF on test set ===
y_pred_rf = grid_rf.predict(X_test)
print("\n=== RandomForest Test Accuracy ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
