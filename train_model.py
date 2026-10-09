import os
import math
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import clean_text

# CREATE MODELS FOLDER
os.makedirs('models', exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv('model/UpdatedResumeDataSet.csv')

# CLEAN DATA
df['cleaned_resume'] = df['Resume'].apply(clean_text)

# FEATURES AND LABELS
X = df['cleaned_resume']
y = df['Category']

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# CLASSIFIER PIPELINE
# ============================================================

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.4f}')

# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

report = classification_report(y_test, predictions)
print(report)

with open('models/classification_report.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.4f}\n\n')
    f.write(report)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

labels = sorted(y.unique())
cm = confusion_matrix(y_test, predictions, labels=labels)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
plt.close()

print('Confusion matrix saved to models/confusion_matrix.png')

# SAVE MODEL
joblib.dump(pipeline, 'models/trained_model.pkl')
print('Model saved successfully')


# ============================================================
# BUILD FROZEN REFERENCE STATISTICS FOR THE SIMILARITY ENGINE
#
# These are trained ONCE on the full resume corpus and reused
# for every app session. Because app.py only ever calls
# .transform() on this vectorizer (never .fit_transform() on
# the uploaded resumes), cosine/euclidean/manhattan similarity
# scores stop shifting whenever resumes are added or removed.
# ============================================================

print('Building frozen reference vectorizer for similarity scoring...')

reference_corpus = df['cleaned_resume'].tolist()

reference_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
reference_vectorizer.fit(reference_corpus)

joblib.dump(reference_vectorizer, 'models/tfidf_vectorizer.pkl')
print('Saved models/tfidf_vectorizer.pkl')

# ------------------------------------------------------------
# BUILD FROZEN BM25 IDF STATISTICS
# ------------------------------------------------------------

print('Building frozen BM25 statistics...')

tokenized_corpus = [doc.split() for doc in reference_corpus]
N = len(tokenized_corpus)
doc_lengths = [len(doc) for doc in tokenized_corpus]
avgdl = sum(doc_lengths) / N if N else 0.0

doc_freq = {}
for doc in tokenized_corpus:
    for term in set(doc):
        doc_freq[term] = doc_freq.get(term, 0) + 1

# Lucene-style idf smoothing: always non-negative, so the
# theoretical-max normalisation used at query time can never
# divide by (or produce) a negative number.
idf = {
    term: math.log(1 + (N - freq + 0.5) / (freq + 0.5))
    for term, freq in doc_freq.items()
}

# Fallback idf for a query term never seen in the reference
# corpus (freq treated as 0) -> highest possible idf.
idf_unseen = math.log(1 + (N + 0.5) / 0.5)

bm25_stats = {
    'idf': idf,
    'idf_unseen': idf_unseen,
    'avgdl': avgdl,
    'N': N,
    'k1': 1.5,
    'b': 0.75,
}

joblib.dump(bm25_stats, 'models/bm25_stats.pkl')
print('Saved models/bm25_stats.pkl')

print('\nAll reference artifacts built. Similarity scores will now be stable.')