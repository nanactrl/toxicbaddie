import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(r"C:\FYPPP2\toxicbaddie\data\combined_cleaned_data.csv")

df.columns = df.columns.str.strip()
df = df.dropna(subset=['clean_text', 'label'])
df['clean_text'] = df['clean_text'].astype(str)

X = df['clean_text']
y = df['label']

# -------------------------
# TRAIN-TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# TF-IDF VECTORIZER
# -------------------------
tfidf = TfidfVectorizer(
    max_features=10000,     # Increased for better performance
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF Features: {X_train_tfidf.shape[1]:,}\n")

# =====================================================
# 1. NAIVE BAYES
# =====================================================
print("Training Multinomial Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)

print("\n==============================")
print("1. TF-IDF + NAIVE BAYES")
print("==============================")
print("Accuracy :", accuracy_score(y_test, nb_pred))
print("F1-Score :", f1_score(y_test, nb_pred, average='weighted'))
print(classification_report(y_test, nb_pred))

# =====================================================
# 2. LINEAR SVC (Recommended)
# =====================================================
print("\nTraining LinearSVC...")
lsvc = LinearSVC(C=1.0, dual=False, max_iter=2000, random_state=42)
lsvc.fit(X_train_tfidf, y_train)
lsvc_pred = lsvc.predict(X_test_tfidf)

print("\n==============================")
print("2. TF-IDF + LinearSVC")
print("==============================")
print("Accuracy :", accuracy_score(y_test, lsvc_pred))
print("F1-Score :", f1_score(y_test, lsvc_pred, average='weighted'))
print(classification_report(y_test, lsvc_pred))

# =====================================================
# 3. LOGISTIC REGRESSION
# =====================================================
print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    C=1.0,                    # Regularization strength
    max_iter=2000, 
    random_state=42,
    solver='liblinear'        # Good for sparse data
    # You can also try solver='lbfgs' or 'saga'
)

lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)

print("\n==============================")
print("3. TF-IDF + Logistic Regression")
print("==============================")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print("F1-Score :", f1_score(y_test, lr_pred, average='weighted'))
print(classification_report(y_test, lr_pred))

# =====================================================
# FINAL COMPARISON TABLE
# =====================================================
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"{'Model':<25} {'Accuracy':<10} {'F1-Score':<10}")
print("-" * 60)
print(f"{'Naive Bayes':<25} {accuracy_score(y_test, nb_pred):.4f}     {f1_score(y_test, nb_pred, average='weighted'):.4f}")
print(f"{'LinearSVC':<25} {accuracy_score(y_test, lsvc_pred):.4f}     {f1_score(y_test, lsvc_pred, average='weighted'):.4f}")
print(f"{'Logistic Regression':<25} {accuracy_score(y_test, lr_pred):.4f}     {f1_score(y_test, lr_pred, average='weighted'):.4f}")

