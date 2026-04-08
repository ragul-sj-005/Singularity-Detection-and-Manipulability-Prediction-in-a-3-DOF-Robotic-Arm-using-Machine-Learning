import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\sjrag\OneDrive\Desktop\Machine_Learning_Project_Nag_Rag\create_dataset_nag_rag.csv")

print("Missing values in dataset:")
print(df.isnull().sum())

# -----------------------------
# Input features
# -----------------------------
X = df[["theta1", "theta2", "theta3"]]

# Targets
y_reg = df["manipulability"]
y_cls = df["singularity"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

y_train_cls = y_cls.loc[y_train_reg.index]
y_test_cls = y_cls.loc[y_test_reg.index]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# SVR REGRESSION (Hyperparameter Optimization)
# ======================================================

print("\nOptimizing SVR model...")

svr = SVR(kernel="rbf")

svr_params = {
    "C": [1, 10, 50],
    "gamma": ["scale", 0.1, 0.01],
    "epsilon": [0.1, 0.5]
}

svr_grid = GridSearchCV(
    svr,
    svr_params,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

svr_grid.fit(X_train, y_train_reg)

svr_model = svr_grid.best_estimator_

print("Best SVR Parameters:", svr_grid.best_params_)

# Prediction
y_pred_reg = svr_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print("\nSVR Regression Results")
print("MSE:", mse)
print("R2 Score:", r2)
print("MAE:", mae)

# ======================================================
# SVM CLASSIFICATION (Hyperparameter Optimization)
# ======================================================

print("\nOptimizing SVM model...")

svc = SVC(class_weight="balanced")

svc_params = {
    "C": [0.1, 1, 10, 50],
    "gamma": ["scale", 0.1, 0.01],
    "kernel": ["rbf"]
}

svc_grid = GridSearchCV(
    svc,
    svc_params,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)


svc_grid.fit(X_train, y_train_cls)

svm_model = svc_grid.best_estimator_

print("Best SVM Parameters:", svc_grid.best_params_)

# Prediction
y_pred_cls = svm_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test_cls, y_pred_cls)

print("\nSVM Classification Results")
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_cls, y_pred_cls))

print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_cls))

# ======================================================
# SAVE MODELS
# ======================================================

joblib.dump(svr_model, "manipulability_svr_model.pkl")
joblib.dump(svm_model, "singularity_svm_model.pkl")
joblib.dump(scaler, "theta_scaler.pkl")

print("\nModels saved successfully!")

print("Saved files:")
print("manipulability_svr_model.pkl")
print("singularity_svm_model.pkl")
print("theta_scaler.pkl")