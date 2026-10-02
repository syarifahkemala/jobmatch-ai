from src.recommender import recommend_roles
from pypdf import PdfReader
import streamlit as st

st.set_page_config(
    page_title="JobMatch AI",
    page_icon="💼"
)

st.title("💼 JobMatch AI")
st.write("AI-powered CV and job role matching system.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload your CV",
    type=["pdf"]
)

if uploaded_file:
    reader = PdfReader(uploaded_file)

    cv_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            cv_text += page_text + "\n"

    st.subheader("Extracted CV")
    st.text_area(
        "CV Text",
        cv_text,
        height=300
    )

    predicted_category, results = recommend_roles(cv_text)

    st.subheader("Predicted Category")
    st.write(predicted_category)

    st.subheader("Recommended Roles")

    st.dataframe(
        results[
            [
                "role",
                "similarity",
                "skill_match",
                "category_match",
                "final_score"
            ]
        ]
    )