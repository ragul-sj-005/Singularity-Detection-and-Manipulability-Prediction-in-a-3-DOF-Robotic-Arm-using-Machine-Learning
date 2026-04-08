r"""
import numpy as np
import joblib
import pandas as pd

# -----------------------------
# Load trained models
# -----------------------------
svr_model = joblib.load(r"C:\Users\sjrag\OneDrive\Desktop\Machine_Learning_Project_Nag_Rag\manipulability_svr_model.pkl")
svm_model = joblib.load(r"C:\Users\sjrag\OneDrive\Desktop\Machine_Learning_Project_Nag_Rag\singularity_svm_model.pkl")
scaler = joblib.load(r"C:\Users\sjrag\OneDrive\Desktop\Machine_Learning_Project_Nag_Rag\theta_scaler.pkl")


# Input
theta1 = float(input("Enter theta1: "))
theta2 = float(input("Enter theta2: "))
theta3 = float(input("Enter theta3: "))

theta_input = [theta1, theta2, theta3]

# Convert to DataFrame (IMPORTANT)
theta_df = pd.DataFrame([theta_input], columns=["theta1", "theta2", "theta3"])

# Scale
theta_scaled = scaler.transform(theta_df)

# Predictions
manipulability = svr_model.predict(theta_scaled)[0]
singularity = svm_model.predict(theta_scaled)[0]

# Fix invalid values
manipulability = max(0, manipulability)

# Safety correction
THRESHOLD = 20

# ML-based correction
if manipulability < THRESHOLD:
    singularity = 1

# Physics-based correction (VERY IMPORTANT)
if abs(theta2) < 5 and abs(theta3) < 5:
    print("⚠️ Near straight-line configuration detected")
    singularity = 1
    manipulability = 0

# Output
print("\n===== RESULTS =====")
print(f"Manipulability: {manipulability:.2f}")

if singularity == 1:
    print("⚠️ Configuration is SINGULAR")
else:
    print("✅ Configuration is SAFE")
"""

"""
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load trained models
# -----------------------------
svr_model = joblib.load("C:/Users/sjrag/OneDrive/Desktop/Machine_Learning_Project_Nag_Rag/manipulability_svr_model.pkl")
svm_model = joblib.load("C:/Users/sjrag/OneDrive/Desktop/Machine_Learning_Project_Nag_Rag/singularity_svm_model.pkl")
scaler = joblib.load("C:/Users/sjrag/OneDrive/Desktop/Machine_Learning_Project_Nag_Rag/theta_scaler.pkl")

# -----------------------------
# Generate random samples
# -----------------------------
samples = 10000

data = []

for _ in range(samples):

    theta1 = np.random.uniform(-180, 180)
    theta2 = np.random.uniform(-180, 180)
    theta3 = np.random.uniform(-180, 180)

    theta_df = pd.DataFrame([[theta1, theta2, theta3]],
                            columns=["theta1", "theta2", "theta3"])

    theta_scaled = scaler.transform(theta_df)

    # Predictions
    manipulability = svr_model.predict(theta_scaled)[0]
    singularity = svm_model.predict(theta_scaled)[0]

    # Fix invalid values
    manipulability = max(0, manipulability)

    # Safety correction
    THRESHOLD = 20

    if manipulability < THRESHOLD:
        singularity = 1

    # Physics-based correction
    if abs(theta2) < 5 and abs(theta3) < 5:
        singularity = 1
        manipulability = 0

    data.append([theta1, theta2, theta3, manipulability, singularity])

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame(data, columns=[
    "theta1", "theta2", "theta3", "manipulability", "singularity"
])

# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(10, 7))

# Scatter plot (theta2 vs theta3)
scatter = plt.scatter(
    df["theta2"],
    df["theta3"],
    c=df["manipulability"],
    cmap="viridis",
    s=5
)

plt.colorbar(scatter, label="Manipulability")

# Highlight singular points
singular_df = df[df["singularity"] == 1]

plt.scatter(
    singular_df["theta2"],
    singular_df["theta3"],
    color="red",
    s=5,
    label="Singular Region"
)

plt.xlabel("Theta2")
plt.ylabel("Theta3")
plt.title("Manipulability & Singular Regions")
plt.legend()

plt.show()
"""

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load trained models
# -----------------------------
svr_model = joblib.load("C:/Users/sjrag/OneDrive/Desktop/Machine_Learning_Project_Nag_Rag/manipulability_svr_model.pkl")
svm_model = joblib.load("C:/Users/sjrag/OneDrive/Desktop/Machine_Learning_Project_Nag_Rag/singularity_svm_model.pkl")
scaler = joblib.load("C:/Users/sjrag/OneDrive/Desktop/Machine_Learning_Project_Nag_Rag/theta_scaler.pkl")

# -----------------------------
# Robot Link Lengths
# -----------------------------
L1 = 7
L2 = 7
L3 = 7

# -----------------------------
# Jacobian Function
# -----------------------------
def jacobian(t1, t2, t3):
    j11 = -(L2*np.cos(t2)+L3*np.cos(t2+t3))*np.sin(t1)
    j12 = -(L2*np.sin(t2)+L3*np.sin(t2+t3))*np.cos(t1)
    j13 = -L3*np.sin(t2+t3)*np.cos(t1)

    j21 = (L2*np.cos(t2)+L3*np.cos(t2+t3))*np.cos(t1)
    j22 = -(L2*np.sin(t2)+L3*np.sin(t2+t3))*np.sin(t1)
    j23 = -L3*np.sin(t2+t3)*np.sin(t1)

    j31 = 0
    j32 = L2*np.cos(t2) + L3*np.cos(t2+t3)
    j33 = L3*np.cos(t2+t3)

    return np.array([
        [j11, j12, j13],
        [j21, j22, j23],
        [j31, j32, j33]
    ])

# -----------------------------
# Generate Samples
# -----------------------------
samples = 10000

theta2_list = []
theta3_list = []

actual_manip_list = []
pred_manip_list = []

actual_sing_list = []
pred_sing_list = []

# -----------------------------
# Loop
# -----------------------------
for _ in range(samples):

    theta1 = np.random.uniform(-180, 180)
    theta2 = np.random.uniform(-180, 180)
    theta3 = np.random.uniform(-180, 180)

    t1 = np.radians(theta1)
    t2 = np.radians(theta2)
    t3 = np.radians(theta3)

    # -----------------------------
    # ACTUAL (Jacobian)
    # -----------------------------
    J = jacobian(t1, t2, t3)
    actual_manip = np.sqrt(abs(np.linalg.det(J @ J.T)))

    # Define actual singularity
    THRESHOLD = 1
    actual_sing = 1 if actual_manip < THRESHOLD else 0

    # -----------------------------
    # ML Prediction (IMPROVED)
    # -----------------------------
    theta_df = pd.DataFrame([[theta1, theta2, theta3]],
                        columns=["theta1", "theta2", "theta3"])

    theta_scaled = scaler.transform(theta_df)

    # Raw predictions
    pred_manip = svr_model.predict(theta_scaled)[0]
    pred_sing = svm_model.predict(theta_scaled)[0]

    # -----------------------------
    # Fix invalid values
    # -----------------------------
    pred_manip = max(0, pred_manip)

    # -----------------------------
    # Safety correction (ML + logic)
    # -----------------------------
    THRESHOLD = 20

    # Low manipulability → singular
    if pred_manip < THRESHOLD and pred_sing == 0:
        pred_sing = 1

    # -----------------------------
    # Physics-based correction (VERY IMPORTANT)
    # -----------------------------
    if abs(theta2) < 5 and abs(theta3) < 5:
        pred_sing = 1
        pred_manip = 0

    # -----------------------------
    # Store
    # -----------------------------
    theta2_list.append(theta2)
    theta3_list.append(theta3)

    actual_manip_list.append(actual_manip)
    pred_manip_list.append(pred_manip)

    actual_sing_list.append(actual_sing)
    pred_sing_list.append(pred_sing)

# -----------------------------
# PLOT 1: Manipulability Comparison
# -----------------------------
plt.figure(figsize=(14, 6))

# ACTUAL
plt.subplot(1, 2, 1)
plt.scatter(theta2_list, theta3_list,
            c=actual_manip_list, cmap='viridis', s=5)
plt.colorbar(label="Actual Manipulability")
plt.title("Actual (Jacobian)")
plt.xlabel("Theta2")
plt.ylabel("Theta3")

# PREDICTED
plt.subplot(1, 2, 2)
plt.scatter(theta2_list, theta3_list,
            c=pred_manip_list, cmap='viridis', s=5)
plt.colorbar(label="Predicted Manipulability")
plt.title("ML Predicted")
plt.xlabel("Theta2")
plt.ylabel("Theta3")

plt.tight_layout()
plt.show()

# -----------------------------
# PLOT 2: Accuracy (Regression)
# -----------------------------
plt.figure(figsize=(8, 6))

plt.scatter(actual_manip_list, pred_manip_list, s=5, alpha=0.5)

max_val = max(max(actual_manip_list), max(pred_manip_list))
plt.plot([0, max_val], [0, max_val], color='red')

plt.xlabel("Actual Manipulability")
plt.ylabel("Predicted Manipulability")
plt.title("Actual vs Predicted")

plt.grid()
plt.show()

# -----------------------------
# PLOT 3: Singularity Comparison
# -----------------------------
plt.figure(figsize=(14, 6))

# ACTUAL singularity
plt.subplot(1, 2, 1)
plt.scatter(theta2_list, theta3_list,
            c=actual_sing_list, cmap='coolwarm', s=5)
plt.title("Actual Singularity (Jacobian)")
plt.xlabel("Theta2")
plt.ylabel("Theta3")

# PREDICTED singularity
plt.subplot(1, 2, 2)
plt.scatter(theta2_list, theta3_list,
            c=pred_sing_list, cmap='coolwarm', s=5)
plt.title("Predicted Singularity (SVM)")
plt.xlabel("Theta2")
plt.ylabel("Theta3")

# -----------------------------
# PLOT 4: Error Map
# -----------------------------
error = np.abs(np.array(actual_manip_list) - np.array(pred_manip_list))

plt.figure(figsize=(7, 6))
plt.scatter(theta2_list, theta3_list,
            c=error, cmap='hot', s=5)

plt.colorbar(label="Absolute Error")
plt.title("Error Distribution (Actual vs ML)")
plt.xlabel("Theta2")
plt.ylabel("Theta3")

plt.tight_layout()
plt.show()