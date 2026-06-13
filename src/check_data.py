import pandas as pd
import os

BASE_DIR = r"C:\FYPPP2\toxicbaddie\data"

aggressive_path = os.path.join(BASE_DIR, "Aggressive_All.csv")
non_aggressive_path = os.path.join(BASE_DIR, "Non_Aggressive_All.csv")

# Load CSV safely
aggressive = pd.read_csv(aggressive_path)
non_aggressive = pd.read_csv(non_aggressive_path)

print("Aggressive shape:", aggressive.shape)
print("Non-aggressive shape:", non_aggressive.shape)

# Add labels (IMPORTANT for ML training)
aggressive["label"] = 1
non_aggressive["label"] = 0

# Combine dataset
df = pd.concat([aggressive, non_aggressive], ignore_index=True)

print("\nCombined shape:", df.shape)
print(df.head())