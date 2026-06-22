import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# =====================================================
# LOAD DATA
# =====================================================
print("Loading dataset...")

df = pd.read_csv(r"C:\FYPPP2\toxicbaddie\data\combined_cleaned_data.csv")

df.columns = df.columns.str.strip()

df = df.dropna(subset=['clean_text', 'label'])
df['clean_text'] = df['clean_text'].astype(str)

X = df['clean_text']
y = df['label']

print(f"Total Records: {len(df):,}")
print("\nClass Distribution:")
print(df['label'].value_counts())

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples: {len(X_train):,}")
print(f"Testing Samples : {len(X_test):,}")

# =====================================================
# TF-IDF VECTORIZATION
# =====================================================
print("\nCreating TF-IDF Features...")

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF Features: {X_train_tfidf.shape[1]:,}")

# =====================================================
# 1. MULTINOMIAL NAIVE BAYES
# =====================================================
print("\nTraining Multinomial Naive Bayes...")

nb = MultinomialNB()

nb.fit(X_train_tfidf, y_train)

nb_pred = nb.predict(X_test_tfidf)

nb_acc = accuracy_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred, average='weighted')

print("\n==============================")
print("1. TF-IDF + Naive Bayes")
print("==============================")
print("Accuracy :", round(nb_acc, 4))
print("F1-Score :", round(nb_f1, 4))
print(classification_report(y_test, nb_pred))

# =====================================================
# 2. OPTIMIZED LINEARSVC
# =====================================================
print("\nTraining Optimized LinearSVC...")

param_grid = {
    'C': [0.1, 0.5, 1, 2, 5, 10, 20],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(
    estimator=LinearSVC(
        dual=False,
        max_iter=10000,
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

grid.fit(X_train_tfidf, y_train)

best_lsvc = grid.best_estimator_

print("\nBest LinearSVC Parameters:")
print(grid.best_params_)

lsvc_pred = best_lsvc.predict(X_test_tfidf)

lsvc_acc = accuracy_score(y_test, lsvc_pred)
lsvc_f1 = f1_score(y_test, lsvc_pred, average='weighted')

print("\n==============================")
print("2. Optimized TF-IDF + LinearSVC")
print("==============================")
print("Accuracy :", round(lsvc_acc, 4))
print("F1-Score :", round(lsvc_f1, 4))
print(classification_report(y_test, lsvc_pred))

# =====================================================
# 3. LOGISTIC REGRESSION
# =====================================================
print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    C=1.0,
    max_iter=5000,
    random_state=42,
    solver='liblinear'
)

lr.fit(X_train_tfidf, y_train)

lr_pred = lr.predict(X_test_tfidf)

lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average='weighted')

print("\n==============================")
print("3. TF-IDF + Logistic Regression")
print("==============================")
print("Accuracy :", round(lr_acc, 4))
print("F1-Score :", round(lr_f1, 4))
print(classification_report(y_test, lr_pred))

# =====================================================
# FINAL COMPARISON
# =====================================================
print("\n")
print("=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(f"{'Model':<30}{'Accuracy':<15}{'Weighted F1':<15}")
print("-" * 65)

print(f"{'Naive Bayes':<30}{nb_acc:.4f}{'':<7}{nb_f1:.4f}")
print(f"{'Optimized LinearSVC':<30}{lsvc_acc:.4f}{'':<7}{lsvc_f1:.4f}")
print(f"{'Logistic Regression':<30}{lr_acc:.4f}{'':<7}{lr_f1:.4f}")

print("=" * 65)

# =====================================================
# BEST MODEL
# =====================================================
results = {
    "Naive Bayes": nb_f1,
    "Optimized LinearSVC": lsvc_f1,
    "Logistic Regression": lr_f1
}

best_model = max(results, key=results.get)

print(f"\nBest Model Based on Weighted F1-Score: {best_model}")
print(f"Best F1-Score: {results[best_model]:.4f}")