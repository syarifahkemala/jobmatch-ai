import joblib
import re
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from supabase import create_client
import os
from dotenv import load_dotenv


# =========================
# 1. Load environment
# =========================

load_dotenv()


# =========================
# 2. Load trained model
# =========================

model = joblib.load("models/logistic_regression.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# =========================
# 3. Load embedding model
# =========================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 4. Connect to Supabase
# =========================

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)


# =========================
# 5. Cleaning function
# =========================

def clean_resume(text):

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s+#.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# 6. Category-role mapping
# =========================

category_role_mapping = {

    "AI Engineer": [
        "INFORMATION-TECHNOLOGY",
        "ENGINEERING"
    ],

    "Data Scientist": [
        "INFORMATION-TECHNOLOGY",
        "ENGINEERING",
        "CONSULTANT"
    ],

    "Backend Engineer": [
        "INFORMATION-TECHNOLOGY",
        "ENGINEERING"
    ],

    "Admin": [
        "HR",
        "BUSINESS-DEVELOPMENT"
    ],

    "HR": [
        "HR",
        "BUSINESS-DEVELOPMENT"
    ],

    "Finance": [
        "FINANCE",
        "ACCOUNTANT",
        "BANKING"
    ]
}


# =========================
# 7. Skill matching
# =========================

def calculate_skill_match(cv_text, required_skills):

    skills = required_skills.split(";")

    matched = []

    for skill in skills:

        if skill.lower() in cv_text.lower():
            matched.append(skill)

    if len(skills) == 0:
        return 0

    return len(matched) / len(skills)


# =========================
# 8. Category matching
# =========================

def calculate_category_match(role, predicted_category):

    categories = category_role_mapping.get(role, [])

    if predicted_category in categories:
        return 1

    else:
        return 0


# =========================
# 9. Recommendation function
# =========================

def recommend_roles(cv_text):

    # 1. Cleaning
    clean_text = clean_resume(cv_text)


    # 2. ML prediction
    cv_tfidf = tfidf.transform([clean_text])

    predicted_category = model.predict(cv_tfidf)[0]


    # 3. Embedding
    cv_embedding = embedding_model.encode(
        clean_text,
        show_progress_bar=False
    )


    # 4. Semantic search
    response = supabase.rpc(
        "match_company_roles",
        {
            "query_embedding": cv_embedding.tolist(),
            "match_count": 6
        }
    ).execute()


    results = pd.DataFrame(response.data)


    # 5. Skill matching
    results["skill_match"] = results.apply(
        lambda row: calculate_skill_match(
            clean_text,
            row["required_skill"]
        ),
        axis=1
    )


    # 6. Category matching
    results["category_match"] = results["role"].apply(
        lambda role: calculate_category_match(
            role,
            predicted_category
        )
    )


    # 7. Final score
    results["final_score"] = (
        0.5 * results["similarity"]
        + 0.3 * results["skill_match"]
        + 0.2 * results["category_match"]
    )


    # 8. Ranking
    results = results.sort_values(
        "final_score",
        ascending=False
    ).reset_index(drop=True)


    return predicted_category, results