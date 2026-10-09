import numpy as np
import pandas as pd


def rank_candidates(
    resume_names,
    cosine_scores,
    jaccard_scores,
    euclidean_scores,
    manhattan_scores,
    bm25_scores,
    skill_match_scores,
    skills_list
):
    cosine_scores = np.array(cosine_scores, dtype=float)
    jaccard_scores = np.array(jaccard_scores, dtype=float)
    euclidean_scores = np.array(euclidean_scores, dtype=float)
    manhattan_scores = np.array(manhattan_scores, dtype=float)
    bm25_scores = np.array(bm25_scores, dtype=float)
    skill_match_scores = np.array(skill_match_scores, dtype=float)

    # All five measures are now guaranteed to be in [0, 1], and
    # these weights sum to 1.0, so this average is also in [0, 1].
    similarity_avg = (
        (0.25 * cosine_scores) +
        (0.15 * jaccard_scores) +
        (0.20 * euclidean_scores) +
        (0.20 * manhattan_scores) +
        (0.20 * bm25_scores)
    )

    # Blend lexical similarity with keyword/skill coverage. Skill
    # coverage is what recruiters usually mean by "ATS score" and
    # keeps results spread across an intuitive range instead of
    # clustering low the way raw TF-IDF overlap tends to.
    final_scores = (0.6 * similarity_avg) + (0.4 * skill_match_scores)

    # Belt-and-suspenders: even if an upstream measure ever misbehaves,
    # the final score (and therefore the ATS %) can never leave [0, 1].
    final_scores = np.clip(final_scores, 0.0, 1.0)

    df = pd.DataFrame({
        'Candidate': resume_names,
        'Cosine Similarity': np.round(cosine_scores, 4),
        'Jaccard Similarity': np.round(jaccard_scores, 4),
        'Euclidean Similarity': np.round(euclidean_scores, 4),
        'Manhattan Similarity': np.round(manhattan_scores, 4),
        'BM25 Similarity': np.round(bm25_scores, 4),
        'Skill Match': np.round(skill_match_scores, 4),
        'Final Score': final_scores,
        'Skills': skills_list
    })

    df = df.sort_values(by='Final Score', ascending=False).reset_index(drop=True)

    df.index = df.index + 1
    df.insert(0, 'Rank', df.index)

    return df