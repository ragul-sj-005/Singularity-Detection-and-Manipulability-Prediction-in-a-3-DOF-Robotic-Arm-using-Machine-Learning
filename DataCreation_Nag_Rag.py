import numpy as np
import pandas as pd
import os

# -----------------------------
# Robot Link Lengths
# -----------------------------
L1 = 7
L2 = 7
L3 = 7

# -----------------------------
# Jacobian Matrix
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

    J = np.array([
        [j11, j12, j13],
        [j21, j22, j23],
        [j31, j32, j33]
    ])

    return J


# -----------------------------
# Dataset Generation
# -----------------------------

samples = 100000
data = []

for i in range(samples):

    theta1 = np.random.uniform(-180, 180)
    theta2 = np.random.uniform(-180, 180)
    theta3 = np.random.uniform(-180, 180)

    t1 = np.radians(theta1)
    t2 = np.radians(theta2)
    t3 = np.radians(theta3)

    J = jacobian(t1,t2,t3)

    manipulability = np.sqrt(abs(np.linalg.det(J @ J.T)))
    
    detJ = abs(np.linalg.det(J))

    if detJ < 5:
        singularity = 1
    else:
        singularity = 0

    data.append([
        theta1,
        theta2,
        theta3,
        detJ,
        manipulability,
        singularity,
    ])

# -----------------------------
# Save Dataset
# -----------------------------

columns = [
    "theta1",
    "theta2",
    "theta3",
    "detJ",
    "manipulability",
    "singularity"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("create_dataset_nag_rag.csv", index=False)

print("Dataset generated successfully")
print("Total samples:", len(df))

print("File saved at:", os.getcwd())