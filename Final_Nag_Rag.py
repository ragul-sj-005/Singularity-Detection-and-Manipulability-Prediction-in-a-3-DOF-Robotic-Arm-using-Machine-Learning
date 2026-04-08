import numpy as np
import joblib
import pandas as pd
import serial
import time

# -----------------------------
# Serial Setup
# -----------------------------
arduino = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)

# -----------------------------
# Load Models
# -----------------------------
svr_model = joblib.load(r"C:\Users\sjrag\OneDrive\Desktop\Machine_Learning_Project_Nag_Rag\manipulability_svr_model.pkl")
svm_model = joblib.load(r"C:\Users\sjrag\OneDrive\Desktop\Machine_Learning_Project_Nag_Rag\singularity_svm_model.pkl")
scaler = joblib.load(r"C:\Users\sjrag\OneDrive\Desktop\Machine_Learning_Project_Nag_Rag\theta_scaler.pkl")

# -----------------------------
# Robot Link Lengths
# -----------------------------
L1 = 7
L2 = 7
L3 = 7

# -----------------------------
# Forward Kinematics
# -----------------------------
def forward_kinematics(t1, t2, t3):
    x = (L2*np.cos(t2) + L3*np.cos(t2+t3)) * np.cos(t1)
    y = (L2*np.cos(t2) + L3*np.cos(t2+t3)) * np.sin(t1)
    z = L1 + L2*np.sin(t2) + L3*np.sin(t2+t3)
    return x, y, z

# -----------------------------
# NORMALIZATION FUNCTION 
# -----------------------------
def normalize(theta):
    return theta / 2   # -180 → -90

# -----------------------------
# USER INPUT
# -----------------------------
theta1 = float(input("Enter theta1 (-180 to 180): "))
theta2 = float(input("Enter theta2 (-180 to 180): "))
theta3 = float(input("Enter theta3 (-180 to 180): "))

# -----------------------------
# ML Prediction
# -----------------------------
theta_df = pd.DataFrame([[theta1, theta2, theta3]],
                        columns=["theta1","theta2","theta3"])

theta_scaled = scaler.transform(theta_df)

manipulability = svr_model.predict(theta_scaled)[0]
singularity = svm_model.predict(theta_scaled)[0]

manipulability = max(0, manipulability)

# -----------------------------
# Safety Logic
# -----------------------------
if manipulability < 20:
    singularity = 1

if abs(theta2) < 5 and abs(theta3) < 5:
    print("⚠️ Near straight-line configuration")
    singularity = 1
    manipulability = 0

# -----------------------------
# Forward Kinematics
# -----------------------------

#  NORMALIZE INSTEAD OF CLIP
theta1 = normalize(theta1)
theta2 = normalize(theta2)
theta3 = normalize(theta3)

t1_rad = np.radians(theta1)
t2_rad = np.radians(theta2)
t3_rad = np.radians(theta3)

x, y, z = forward_kinematics(t1_rad, t2_rad, t3_rad)

# -----------------------------
# Convert to Servo Angles
# -----------------------------
t1 = int(theta1 + 90)
t2 = int(theta2 + 90)
t3 = int(theta3 + 90)

# -----------------------------
# SEND ONLY TARGET (Arduino handles return)
# -----------------------------
if (singularity != 1):
    arduino.write(f"{t1},{t2},{t3}\n".encode())     

# -----------------------------
# OUTPUT
# -----------------------------
print("\n===== RESULTS =====")
print(f"Manipulability: {manipulability:.2f}")

if singularity == 1:
    print("⚠️ SINGULAR")
else:
    print("✅ SAFE")

print("\nPosition:")
print(f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}")