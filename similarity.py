import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# JACCARD SIMILARITY
# Purely set-based and computed independently for each resume,
# so it was already stable — kept as-is.
# ============================================================

def jaccard_similarity(text1, text2):
    set1 = set(preprocess_text(text1).split())
    set2 = set(preprocess_text(text2).split())

    union = set1.union(set2)
    if len(union) == 0:
        return 0.0

    intersection = set1.intersection(set2)
    return len(intersection) / len(union)


def calculate_jaccard_scores(job_desc, resumes):
    job_desc = preprocess_text(job_desc)
    return np.array([jaccard_similarity(job_desc, r) for r in resumes])


# ============================================================
# TF-IDF BASED SIMILARITIES (cosine / euclidean / manhattan)
#
# IMPORTANT: `vectorizer` must be the FROZEN vectorizer trained
# once on a large reference corpus (see reference_builder /
# train_model.py) and loaded with joblib.load(). We only ever
# call .transform() here, never .fit_transform() on the live
# upload batch — that is what makes scores stop changing when
# resumes are added or removed.
# ============================================================

def _vectorize(vectorizer, job_desc, resumes):
    job_vec = vectorizer.transform([preprocess_text(job_desc)])
    resume_vecs = vectorizer.transform([preprocess_text(r) for r in resumes])
    return job_vec, resume_vecs


def calculate_cosine_similarity(job_desc, resumes, vectorizer):
    job_vec, resume_vecs = _vectorize(vectorizer, job_desc, resumes)
    return cosine_similarity(job_vec, resume_vecs)[0]


def calculate_euclidean_similarity(job_desc, resumes, vectorizer):
    job_vec, resume_vecs = _vectorize(vectorizer, job_desc, resumes)
    job_arr = job_vec.toarray()[0]

    scores = []
    for row in resume_vecs.toarray():
        distance = np.linalg.norm(job_arr - row)
        scores.append(1 / (1 + distance))

    return np.array(scores)


def calculate_manhattan_similarity(job_desc, resumes, vectorizer):
    job_vec, resume_vecs = _vectorize(vectorizer, job_desc, resumes)
    job_arr = job_vec.toarray()[0]

    scores = []
    for row in resume_vecs.toarray():
        distance = np.sum(np.abs(job_arr - row))
        scores.append(1 / (1 + distance))

    return np.array(scores)


# ============================================================
# BM25 — reimplemented with FROZEN idf statistics
#
# rank_bm25's BM25Okapi has two problems for this use case:
#   1. It rebuilds its IDF table from whatever resumes are in
#      the current batch -> scores shift as resumes are added.
#   2. get_scores() returns an UNBOUNDED raw score (can be 0,
#      8, 40...) which is why your ATS blew past 100 once it
#      was averaged in with 0-1 scores.
#
# Fix: idf/avgdl come from `bm25_stats`, precomputed ONCE on a
# large fixed reference corpus (see train_model.py) and loaded
# at runtime. Each resume's score only depends on the job
# description and that resume — never on other uploads. We
# also normalise by the query's theoretical maximum score, so
# the result is always in [0, 1].
# ============================================================

def _bm25_score_single(query_tokens, doc_tokens, idf, idf_unseen, avgdl, k1, b):
    doc_len = len(doc_tokens)

    freqs = {}
    for t in doc_tokens:
        freqs[t] = freqs.get(t, 0) + 1

    score = 0.0
    theoretical_max = 0.0

    for term in set(query_tokens):
        term_idf = idf.get(term, idf_unseen)
        # As term frequency -> infinity, the saturation term ->
        # (k1 + 1). Summing that over query terms gives the
        # highest score this query could ever produce.
        theoretical_max += term_idf * (k1 + 1)

        f = freqs.get(term, 0)
        if f == 0:
            continue

        denom = f + k1 * (1 - b + b * (doc_len / avgdl if avgdl else 1))
        score += term_idf * (f * (k1 + 1)) / denom

    if theoretical_max <= 0:
        return 0.0

    return float(min(score / theoretical_max, 1.0))


def calculate_bm25_similarity(job_desc, resumes, bm25_stats):
    idf = bm25_stats['idf']
    idf_unseen = bm25_stats['idf_unseen']
    avgdl = bm25_stats['avgdl']
    k1 = bm25_stats.get('k1', 1.5)
    b = bm25_stats.get('b', 0.75)

    query_tokens = preprocess_text(job_desc).split()

    scores = []
    for resume in resumes:
        doc_tokens = preprocess_text(resume).split()
        scores.append(
            _bm25_score_single(query_tokens, doc_tokens, idf, idf_unseen, avgdl, k1, b)
        )

    return np.array(scores)


# ============================================================
# SKILL MATCH SCORE
# What recruiters usually mean in practice by "ATS score":
# what fraction of the skills the JD asks for actually show up
# in the resume. This is what pulls a genuinely strong match up
# into the 80-90% range instead of the 30-50% range that pure
# lexical/TF-IDF overlap tends to produce.
# ============================================================

def calculate_skill_match(job_skills, resume_skills_list):
    job_set = set(s.lower() for s in job_skills)

    scores = []
    for resume_skills in resume_skills_list:
        if not job_set:
            scores.append(0.0)
            continue
        r_set = set(s.lower() for s in resume_skills)
        scores.append(len(job_set & r_set) / len(job_set))

    return np.array(scores)