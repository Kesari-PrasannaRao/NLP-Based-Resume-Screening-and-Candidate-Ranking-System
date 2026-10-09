import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    filtered_words = []

    for word in words:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            filtered_words.append(lemma)

    return ' '.join(filtered_words)

SKILLS_DB = [
    # --- Programming / IT ---
    "python", "java", "c++", "c#", "javascript", "typescript", "php", "ruby",
    "go", "golang", "kotlin", "swift", "r programming", "scala", "matlab",
    "sql", "nosql", "mysql", "postgresql", "mongodb", "oracle", "sqlite",
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
    "spring boot", "hibernate", "rest api", "graphql", "microservices",
    "git", "github", "gitlab", "bitbucket", "linux", "shell scripting",
    "bash", "unit testing", "agile", "scrum", "jira", "object oriented programming",

    # --- Data / ML / AI ---
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "data engineering",
    "data visualization", "statistics", "pandas", "numpy", "scikit learn",
    "tensorflow", "pytorch", "keras", "spacy", "nltk", "opencv",
    "power bi", "tableau", "excel", "big data", "hadoop", "spark", "etl",
    "airflow", "feature engineering", "a/b testing", "predictive modeling",

    # --- Cloud / DevOps ---
    "aws", "azure", "google cloud", "gcp", "docker", "kubernetes",
    "terraform", "jenkins", "ci/cd", "devops", "ansible", "cloud computing",
    "monitoring", "linux administration", "networking",

    # --- Web / Design ---
    "ui/ux design", "ui design", "ux design", "figma", "adobe xd",
    "sketch", "photoshop", "illustrator", "responsive design", "web design",
    "wordpress", "seo", "graphic design", "prototyping", "wireframing",
    "typography", "branding",

    # --- Business / Finance / Ops ---
    "project management", "product management", "business analysis",
    "financial analysis", "accounting", "budgeting", "forecasting",
    "auditing", "taxation", "risk management", "supply chain management",
    "procurement", "inventory management", "operations management",
    "stakeholder management", "vendor management", "erp", "sap", "quickbooks",

    # --- Sales / Marketing ---
    "digital marketing", "content marketing", "social media marketing",
    "email marketing", "market research", "brand management",
    "sales strategy", "lead generation", "crm", "salesforce",
    "google analytics", "google ads", "copywriting", "public relations",

    # --- HR ---
    "recruitment", "talent acquisition", "employee relations",
    "performance management", "hr operations", "onboarding",
    "conflict resolution", "compensation and benefits", "payroll",
    "training and development", "workforce planning",

    # --- Healthcare ---
    "patient care", "clinical research", "medical coding", "nursing",
    "pharmacology", "healthcare management", "electronic health records",

    # --- Legal / Education ---
    "contract negotiation", "legal research", "compliance",
    "curriculum development", "classroom management", "lesson planning",

    # --- General / Soft skills ---
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "time management", "adaptability", "creativity",
    "negotiation", "presentation skills", "decision making", "mentoring",
    "customer service", "multitasking", "attention to detail",
]

_SKILL_PATTERNS = [
    (skill, re.compile(r'\b' + re.escape(skill) + r'\b'))
    for skill in SKILLS_DB
]


def extract_skills(text):
    """
    Matches against the master skill taxonomy using word-boundary
    regex (safe against partial-word false positives) on the raw
    lowercased text. Works for resumes AND job descriptions.
    """
    text_lower = " " + re.sub(r'[^a-z0-9\s\+\#\./]', ' ', str(text).lower()) + " "

    found = []
    for skill, pattern in _SKILL_PATTERNS:
        if pattern.search(text_lower):
            found.append(skill)

    return sorted(set(found))