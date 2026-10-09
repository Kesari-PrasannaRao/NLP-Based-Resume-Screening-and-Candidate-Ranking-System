
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter

from preprocess import clean_text, extract_skills

from similarity import (
    calculate_cosine_similarity,
    calculate_jaccard_scores,
    calculate_euclidean_similarity,
    calculate_manhattan_similarity,
    calculate_bm25_similarity,
    calculate_skill_match
)

from ranking import rank_candidates

from utils import (
    extract_text_from_pdf,
    extract_text_from_docx
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #EEF4FF;
    }

    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #0B3D91;
    }

    .subtitle {
        color: grey;
        font-size: 18px;
    }

    section[data-testid="stSidebar"] {
        background: #123A75;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .sidebar-btn {
        display: block;
        width: 100%;
        box-sizing: border-box;
        background: #2563EB;
        color: white !important;
        text-decoration: none !important;
        padding: 13px 15px;
        margin: 8px 0;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        text-align: left;
        transition: all 0.2s ease;
    }

    .sidebar-btn:hover {
        background: #1D4ED8;
        color: white !important;
        transform: translateX(3px);
    }

    .stButton > button {
        width: 100%;
        background: #2563EB;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        border: none;
    }

    .stButton > button:hover {
        background: #1D4ED8;
    }

    div[data-testid="stMetric"] {
        background: white;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 15px;
    }

    textarea {
        border-radius: 12px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL + FROZEN REFERENCE ARTIFACTS
# ============================================================

# tfidf_vectorizer.pkl and bm25_stats.pkl are built ONCE by
# train_model.py from the full resume dataset.
# They are only transformed here and never re-fitted.

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/trained_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    bm25_stats = joblib.load("models/bm25_stats.pkl")

    return model, vectorizer, bm25_stats


model, tfidf_vectorizer, bm25_stats = load_artifacts()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/240/artificial-intelligence.png",
        width=100
    )

    st.title("AI Recruiter")

    st.markdown("---")

    st.markdown(
        """
        <a href="#home" class="sidebar-btn">🏠 Home</a>

        <a href="#resume-screening" class="sidebar-btn">
            📄 Resume Screening
        </a>

        <a href="#candidate-ranking" class="sidebar-btn">
            🏆 Candidate Ranking
        </a>

        <a href="#analytics" class="sidebar-btn">
            📊 Analytics
        </a>

        <a href="#reports" class="sidebar-btn">
            📥 Reports
        </a>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.success("NLP + Machine Learning")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<div id='home'></div>",
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 6])

with col1:
    st.image(
        "https://img.icons8.com/color/240/resume.png",
        width=100
    )

with col2:
    st.markdown(
        "<div class='main-title'>"
        "NLP-Based Resume Screening & Candidate Ranking System"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>"
        "Intelligent Resume Analysis using NLP & Machine Learning"
        "</div>",
        unsafe_allow_html=True
    )

st.markdown("---")


# ============================================================
# DASHBOARD METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    total_resume_metric = st.empty()

with m2:
    top_score_metric = st.empty()

with m3:
    average_metric = st.empty()

with m4:
    shortlisted_metric = st.empty()


total_resume_metric.metric(
    "📄 Total Resumes",
    0
)

top_score_metric.metric(
    "🏆 Top Score",
    "--"
)

average_metric.metric(
    "📊 Average ATS",
    "--"
)

shortlisted_metric.metric(
    "✅ Selected",
    "--"
)

st.markdown("---")


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.markdown(
    "<div id='resume-screening'></div>",
    unsafe_allow_html=True
)

st.subheader("📝 Job Description")

job_description = st.text_area(
    "Paste the complete Job Description here",
    height=220,
    placeholder="""
Example:

Software Developer

Required Skills

Python
SQL
Machine Learning
AWS
Git
Communication
"""
)

st.markdown("---")


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📂 Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF or DOCX resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

analyze = st.button("🚀 Analyze Resumes")


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if job_description == "":
        st.warning("Please enter Job Description.")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload resumes.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    resume_texts = []
    resume_names = []
    predictions = []
    skills_list = []

    total = len(uploaded_files)

    # ========================================================
    # PROCESS RESUMES
    # ========================================================

    for index, uploaded_file in enumerate(uploaded_files):

        status.info(
            f"Processing {uploaded_file.name}"
        )

        progress.progress(
            (index + 1) / total
        )

        file_name = uploaded_file.name
        temp_path = "temp_" + file_name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        if file_name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(temp_path)
        else:
            text = extract_text_from_docx(temp_path)

        cleaned = clean_text(text)

        resume_texts.append(cleaned)
        resume_names.append(file_name)

        category = model.predict([cleaned])[0]
        predictions.append(category)

        # Extract skills from RAW text so that
        # multi-word skills such as "machine learning"
        # and skills such as "c++" remain intact.

        skills = extract_skills(text)
        skills_list.append(", ".join(skills))

        os.remove(temp_path)

    status.success(
        "All resumes processed successfully!"
    )


    # ========================================================
    # CLEAN JOB DESCRIPTION + EXTRACT REQUIRED SKILLS
    # ========================================================

    cleaned_job = clean_text(job_description)

    job_skills = extract_skills(
        job_description
    )


    # ========================================================
    # SIMILARITY CALCULATIONS
    # ========================================================

    # Cosine / Euclidean / Manhattan use
    # the FROZEN TF-IDF vectorizer.

    # BM25 uses the FROZEN IDF statistics.

    # Every similarity is calculated per resume
    # against the job description.

    cosine_scores = calculate_cosine_similarity(
        cleaned_job,
        resume_texts,
        tfidf_vectorizer
    )

    jaccard_scores = calculate_jaccard_scores(
        cleaned_job,
        resume_texts
    )

    euclidean_scores = calculate_euclidean_similarity(
        cleaned_job,
        resume_texts,
        tfidf_vectorizer
    )

    manhattan_scores = calculate_manhattan_similarity(
        cleaned_job,
        resume_texts,
        tfidf_vectorizer
    )

    bm25_scores = calculate_bm25_similarity(
        cleaned_job,
        resume_texts,
        bm25_stats
    )

    resume_skills_lists = [
        s.split(", ") if s else []
        for s in skills_list
    ]

    skill_match_scores = calculate_skill_match(
        job_skills,
        resume_skills_lists
    )


    # ========================================================
    # RANK CANDIDATES
    # ========================================================

    result_df = rank_candidates(
        resume_names,
        cosine_scores,
        jaccard_scores,
        euclidean_scores,
        manhattan_scores,
        bm25_scores,
        skill_match_scores,
        skills_list
    )

    result_df["Predicted Category"] = predictions


    # ========================================================
    # ATS SCORE
    # ========================================================

    # Final Score is expected to be in [0, 1].
    # Therefore ATS Score is converted to [0, 100].

    result_df["ATS Score"] = (
        result_df["Final Score"]
        .clip(lower=0, upper=1)
        * 100
    ).round(2)


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendations = []
    strengths = []
    interview_status = []

    for score in result_df["Final Score"]:

        if score >= 0.70:

            recommendations.append(
                "⭐⭐⭐ Highly Recommended"
            )

            strengths.append(
                "Excellent"
            )

            interview_status.append(
                "Selected"
            )

        elif score >= 0.60:

            recommendations.append(
                "⭐⭐ Recommended"
            )

            strengths.append(
                "Good"
            )

            interview_status.append(
                "Shortlisted"
            )

        elif score >= 0.50:

            recommendations.append(
                "⭐ Consider"
            )

            strengths.append(
                "Average"
            )

            interview_status.append(
                "Hold"
            )

        else:

            recommendations.append(
                "Not Recommended"
            )

            strengths.append(
                "Poor"
            )

            interview_status.append(
                "Rejected"
            )

    result_df["Resume Strength"] = strengths

    result_df["Recommendation"] = recommendations

    result_df["Interview Status"] = interview_status


    # ========================================================
    # UPDATE DASHBOARD METRICS
    # ========================================================

    total_resume_metric.metric(
        "📄 Total Resumes",
        len(result_df)
    )

    top_score_metric.metric(
        "🏆 Top Score",
        f"{result_df['ATS Score'].max()} %"
    )

    average_metric.metric(
        "📊 Average ATS",
        f"{round(result_df['ATS Score'].mean(), 2)} %"
    )

    shortlisted = len(
        result_df[
            result_df["Interview Status"] == "Selected"
        ]
    )

    shortlisted_metric.metric(
        "✅ Shortlisted",
        shortlisted
    )

    st.markdown("---")


    # ========================================================
    # CANDIDATE RANKING
    # ========================================================

    st.markdown(
        "<div id='candidate-ranking'></div>",
        unsafe_allow_html=True
    )

    st.subheader("🏆 Candidate Ranking")

    styled_df = result_df.style.highlight_max(
        subset=["ATS Score"],
        color="lightgreen"
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")


    # ========================================================
    # TOP CANDIDATE
    # ========================================================

    top_candidate = result_df.iloc[0]

    st.subheader("🥇 Top Candidate")

    c1, c2 = st.columns([1, 3])

    with c1:

        st.image(
            "https://img.icons8.com/color/240/user.png",
            width=170
        )

    with c2:

        st.markdown(
            f"## {top_candidate['Candidate']}"
        )

        st.metric(
            "ATS Score",
            f"{top_candidate['ATS Score']} %"
        )

        st.success(
            top_candidate["Recommendation"]
        )

        st.write(
            f"**Category:** "
            f"{top_candidate['Predicted Category']}"
        )

        st.write(
            f"**Resume Strength:** "
            f"{top_candidate['Resume Strength']}"
        )

        st.write(
            f"**Interview Status:** "
            f"{top_candidate['Interview Status']}"
        )

        st.write("### 🛠 Skills")

        st.info(
            top_candidate["Skills"]
            if top_candidate["Skills"]
            else "No matching skills detected"
        )

    st.markdown("---")


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.markdown(
        "<div id='reports'></div>",
        unsafe_allow_html=True
    )

    st.subheader("📥 Reports")

    csv = result_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Candidate Report",
        csv,
        "Candidate_Ranking.csv",
        "text/csv"
    )

    st.markdown("---")


    # ========================================================
    # ANALYTICS
    # ========================================================

    st.markdown(
        "<div id='analytics'></div>",
        unsafe_allow_html=True
    )

    st.subheader("📊 Recruitment Analytics")

    chart1, chart2 = st.columns(2)

    with chart1:

        fig = px.bar(
            result_df,
            x="Candidate",
            y="ATS Score",
            color="ATS Score",
            text="ATS Score",
            title="Candidate ATS Scores"
        )

        fig.update_layout(
            xaxis_title="Candidate",
            yaxis_title="ATS Score (%)",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with chart2:

        fig = px.pie(
            result_df,
            names="Predicted Category",
            title="Candidate Categories"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")


    # ========================================================
    # SKILL DISTRIBUTION
    # ========================================================

    st.subheader("🛠 Top Skills")

    all_skills = []

    for skill in result_df["Skills"]:

        skill_list = [
            s.strip()
            for s in skill.split(",")
            if s.strip() != ""
        ]

        all_skills.extend(
            skill_list
        )

    if len(all_skills) > 0:

        counter = Counter(
            all_skills
        )

        skill_df = pd.DataFrame(
            {
                "Skill": list(counter.keys()),
                "Count": list(counter.values())
            }
        )

        skill_df = skill_df.sort_values(
            "Count",
            ascending=False
        )

        fig = px.bar(
            skill_df,
            x="Skill",
            y="Count",
            color="Count",
            title="Most Common Skills"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")


    # ========================================================
    # CANDIDATE PROFILES
    # ========================================================

    st.subheader("👥 Candidate Profiles")

    for index, row in result_df.iterrows():

        with st.expander(
            f"📄 {row['Candidate']}"
        ):

            col1, col2 = st.columns([1, 3])

            with col1:

                st.image(
                    "https://img.icons8.com/color/240/user.png",
                    width=120
                )

            with col2:

                st.markdown(
                    f"### {row['Candidate']}"
                )

                st.write(
                    f"**Predicted Category:** "
                    f"{row['Predicted Category']}"
                )

                st.write(
                    f"**ATS Score:** "
                    f"{row['ATS Score']}%"
                )

                st.write(
                    f"**Resume Strength:** "
                    f"{row['Resume Strength']}"
                )

                st.write(
                    f"**Recommendation:** "
                    f"{row['Recommendation']}"
                )

                st.write(
                    f"**Interview Status:** "
                    f"{row['Interview Status']}"
                )

                st.write("### Skills")

                badges = ""

                for skill in row["Skills"].split(","):

                    if skill.strip() != "":

                        badges += f"""
                        <span style="
                            background:#2563EB;
                            color:white;
                            padding:6px 10px;
                            border-radius:8px;
                            margin:4px;
                            display:inline-block;
                        ">
                            {skill.strip()}
                        </span>
                        """

                st.markdown(
                    badges,
                    unsafe_allow_html=True
                )

                st.progress(
                    float(row["Final Score"])
                )

    st.markdown("---")


    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader("📋 Recruitment Summary")

    total_candidates = len(result_df)

    highest_score = result_df["ATS Score"].max()

    average_score = round(
        result_df["ATS Score"].mean(),
        2
    )

    selected_candidates = len(
        result_df[
            result_df["Interview Status"] == "Selected"
        ]
    )

    shortlisted_candidates = len(
        result_df[
            result_df["Interview Status"] == "Shortlisted"
        ]
    )

    hold_candidates = len(
        result_df[
            result_df["Interview Status"] == "Hold"
        ]
    )

    rejected_candidates = len(
        result_df[
            result_df["Interview Status"] == "Rejected"
        ]
    )

    st.info(
        f"""
**Total Candidates:** {total_candidates}

**Highest ATS Score:** {highest_score} %

**Average ATS Score:** {average_score} %

**Selected Candidates:** {selected_candidates}

**Shortlisted Candidates:** {shortlisted_candidates}

**Hold Candidates:** {hold_candidates}

**Rejected Candidates:** {rejected_candidates}
"""
    )

    st.markdown("---")


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <hr>

        <center>

        <h4 style='color:#2563EB;'>
            🤖 NLP-Based Resume Screening and Candidate Ranking System
        </h4>

        Developed using

        <b>
            Python | Streamlit | NLP | Machine Learning | Scikit-learn
        </b>

        </center>
        """,
        unsafe_allow_html=True
    )

