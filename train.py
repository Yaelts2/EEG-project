import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.signal import welch

# === Paths ===
X_CSV = "processed_data/X.csv"
Y_CSV = "processed_data/y.csv"

# === Load data ===
X = pd.read_csv(X_CSV).values
y = pd.read_csv(Y_CSV).values.flatten()

print(f"✅ Loaded data shape: X={X.shape}, y={y.shape}")

# === Label encoding (if not already encoded) ===
# If your y is numeric labels already, skip this step
# Else:
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

# === Feature enhancement: Add frequency band power features ===
# Function to compute band power features per trial
def compute_bandpower_features(eeg_trial, sfreq=100, nperseg=256):
    # eeg_trial: shape (features,), reshape to (channels, time)
    # Your features are statistical, so this may not apply directly.
    # If your raw EEG segments are saved elsewhere, better to extract band power during preprocessing.
    # Here, we assume features represent channels concatenated features; skip this step or
    # implement only if raw data is available.
    return np.array([])  # No additional features here since X is already processed stats

# === For demonstration, just standard scale the features ===
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

# === Classifiers to try ===
rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
svm_clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)

# === Hyperparameter grid for RandomForest ===
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

# === Train and evaluate SVM with default hyperparameters ===
print("\n🔎 Training SVM with RBF kernel...")
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("\n=== SVM Test Accuracy ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# === Optional: Cross-validation scores for both classifiers ===
print("\n🔎 Cross-validation accuracies:")
rf_cv_scores = grid_rf.best_estimator_.score(X_test, y_test)
svm_cv_scores = svm_clf.score(X_test, y_test)
print(f"RandomForest test accuracy: {rf_cv_scores:.4f}")
print(f"SVM test accuracy: {svm_cv_scores:.4f}")
