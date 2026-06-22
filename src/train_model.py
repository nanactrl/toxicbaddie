import os
import joblib
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

data_path = r"C:\FYPPP2\toxicbaddie\data\combined_cleaned_data.csv"

print("Loading cleaned data...")
df = pd.read_csv(data_path)

print(f"\nDataset shape: {df.shape}")
print("\nLabel distribution:")
print(df['label'].value_counts())

print(f"\nNaN in clean_text before cleaning: {df['clean_text'].isna().sum()}")


df = df.dropna(subset=['Message'])

# Replace NaN in clean_text with empty string
df['clean_text'] = df['clean_text'].fillna("").astype(str)

print(f"NaN after cleaning: {df['clean_text'].isna().sum()}")

# ====================== FEATURES & LABELS ======================
X = df["clean_text"]
y = df["label"]

# ====================== TRAIN-TEST SPLIT ======================
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ====================== BUILD PIPELINE ======================
model = Pipeline([
    
    # TF-IDF Feature Extraction
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # Unigram + Bigram
        max_features=50000,
        min_df=3,
        max_df=0.95
    )),
    
    # Linear Support Vector Classifier
    ("clf", LinearSVC(
        C=1.0,
        dual=False,
        class_weight='balanced',
        random_state=42,
        max_iter=5000
    ))
])

# ====================== TRAIN MODEL ======================
print("\nTraining model... Please wait.")

model.fit(X_train, y_train)

print("✅ Model training completed!")

# ====================== PREDICTION ======================
print("\nMaking predictions...")

pred = model.predict(X_test)

# ====================== EVALUATION ======================
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print(f"\nAccuracy Score: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
report = classification_report(y_test, pred)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix:")
print(cm)

# ====================== CROSS VALIDATION ======================
print("\nPerforming Cross Validation...")

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("\nCross Validation Scores:")
print(cv_scores)

print(f"\nAverage CV Accuracy: {cv_scores.mean():.4f}")

# ====================== TF-IDF VOCABULARY SIZE ======================
tfidf = model.named_steps['tfidf']

print(f"\nTF-IDF Vocabulary Size: {len(tfidf.vocabulary_):,}")

# ====================== VISUALIZE CONFUSION MATRIX ======================
print("\nGenerating confusion matrix visualization...")

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

# ====================== SAVE FIGURE ======================
output_dir = r"C:\FYPPP2\toxicbaddie\models"
os.makedirs(output_dir, exist_ok=True)

cm_path = os.path.join(output_dir, "confusion_matrix.png")

plt.savefig(cm_path)
plt.close()

print(f"✅ Confusion matrix saved at:")
print(cm_path)

# ====================== SAVE CLASSIFICATION REPORT ======================
report_path = os.path.join(output_dir, "classification_report.txt")

with open(report_path, "w") as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Accuracy Score: {accuracy:.4f}\n\n")
    
    f.write("Classification Report:\n")
    f.write(report)
    
    f.write("\nCross Validation Scores:\n")
    f.write(str(cv_scores))
    
    f.write(f"\n\nAverage CV Accuracy: {cv_scores.mean():.4f}")

print(f"\n✅ Classification report saved at:")
print(report_path)

# ====================== SAVE MODEL ======================
model_dir = r"C:\FYPPP2\toxicbaddie\models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "toxic_model.pkl")

# Save model only if accuracy is acceptable
if accuracy >= 0.80:
    
    joblib.dump(model, model_path)

    print("\n MODEL SAVED SUCCESSFULLY!")
    print(f"Location: {model_path}")

else:
    print("\n Accuracy below 80%")
    print("Model was NOT saved.")

# ====================== SAVE LABEL MAP ======================
label_map = {
    0: "Non-Toxic",
    1: "Toxic"
}

label_map_path = os.path.join(model_dir, "label_map.pkl")

joblib.dump(label_map, label_map_path)

print("\n Label map saved successfully!")
print(label_map_path)

# ====================== FINAL SUMMARY ======================
print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"""
Final Summary:
--------------
Dataset Size          : {len(df):,}
Training Samples      : {len(X_train):,}
Testing Samples       : {len(X_test):,}
Accuracy Score        : {accuracy:.4f}
Average CV Accuracy   : {cv_scores.mean():.4f}
TF-IDF Vocabulary     : {len(tfidf.vocabulary_):,}
""")

print(" Ready for Discord Bot Deployment!")