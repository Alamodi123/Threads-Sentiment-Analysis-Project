import os
import re
import joblib
import streamlit as st
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from PIL import Image

# Ensure VADER lexicon is available (Streamlit Cloud fix)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# App config
st.set_page_config(
    page_title="Threads Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# Header
st.markdown(
    """
    <h1 style="text-align:center;">💬 Threads Sentiment Analysis</h1>
    <p style="text-align:center; font-size:16px; color:gray;">
    NLP coursework project · TF-IDF + VADER · Classical Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Load models & tools
@st.cache_resource
def load_artifacts():
    return {
        "tfidf": joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")),
        "lr": joblib.load(os.path.join(MODELS_DIR, "logistic_regression_combined.joblib")),
        "svm": joblib.load(os.path.join(MODELS_DIR, "linear_svm_combined.joblib")),
        "nb": joblib.load(os.path.join(MODELS_DIR, "naive_bayes_combined.joblib")),
        "vader": SentimentIntensityAnalyzer(),
    }

artifacts = load_artifacts()

# Text preprocessing (deployment-safe, NO stopwords)
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(text.split())

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["🔮 Predict", "📊 Model Insights", "ℹ️ About & Limitations"]
)

# TAB 1 — PREDICT
with tab1:
    st.markdown("### Enter a Threads review")

    user_text = st.text_area(
        "Type or paste a review:",
        height=130,
        placeholder="e.g. It’s okay, needs some improvements"
    )

    model_choice = st.selectbox(
        "Choose a model:",
        [
            "Linear SVM (recommended)",
            "Logistic Regression",
            "Naive Bayes",
        ]
    )

    st.caption("💡 Linear SVM is more conservative and handles neutral sentiment better.")

    if st.button("Predict Sentiment"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            clean_text = preprocess_text(user_text)
            X_tfidf = artifacts["tfidf"].transform([clean_text])

            vader_scores = artifacts["vader"].polarity_scores(user_text)
            vader_all = np.array([[vader_scores["neg"], vader_scores["neu"],
                                   vader_scores["pos"], vader_scores["compound"]]])
            vader_nb = np.array([[vader_scores["neg"], vader_scores["neu"], vader_scores["pos"]]])

            if model_choice.startswith("Linear"):
                model = artifacts["svm"]
                X = hstack([X_tfidf, vader_all])
                prediction = model.predict(X)[0]

            elif model_choice.startswith("Logistic"):
                model = artifacts["lr"]
                X = hstack([X_tfidf, vader_all])
                prediction = model.predict(X)[0]
                probs = model.predict_proba(X)[0]
                confidence = np.max(probs)

            else:
                model = artifacts["nb"]
                X = hstack([X_tfidf, vader_nb])
                prediction = model.predict(X)[0]

            # Color-coded output
            if prediction == "positive":
                st.success("🟢 **Predicted Sentiment: POSITIVE**")
            elif prediction == "neutral":
                st.info("🔵 **Predicted Sentiment: NEUTRAL**")
            else:
                st.error("🔴 **Predicted Sentiment: NEGATIVE**")

            if model_choice.startswith("Logistic"):
                st.caption(f"Confidence: {confidence:.2f}")

# TAB 2 — MODEL INSIGHTS
with tab2:
    st.markdown("### Model Comparison & Insights")

    st.write(
        """
        This project compares three classical machine learning models trained on
        TF-IDF features combined with lexicon-based VADER sentiment scores.
        """
    )

    with st.expander("📈 Performance Comparison"):
        try:
            img = Image.open(os.path.join(PLOTS_DIR, "accuracy_macro_f1_comparison.png"))
            st.image(img, use_column_width=True)
        except:
            st.info("Performance comparison plot not found.")

    with st.expander("🔍 Model behavior summary"):
        st.write(
            """
            - **Logistic Regression** offers a balance between performance and interpretability.
            - **Linear SVM** is more conservative and performs better on borderline neutral cases.
            - **Naive Bayes** is fast but biased toward majority classes due to independence assumptions.
            """
        )

# TAB 3 — ABOUT & LIMITATIONS
with tab3:
    st.markdown("### About This Project")

    st.write(
        """
        This application was developed as part of a university NLP coursework.
        It demonstrates an end-to-end sentiment analysis pipeline, from data preprocessing
        and feature engineering to model evaluation and interactive deployment.
        """
    )

    st.markdown("### Limitations")
    st.write(
        """
        - Sentiment labels are derived from star ratings and may contain noise.
        - Neutral sentiment is inherently ambiguous and difficult to classify.
        - The system relies solely on textual content without user context.
        """
    )

    st.markdown("### Methods Used")
    st.write(
        """
        - Text preprocessing (normalization)
        - TF-IDF vectorization (unigrams + bigrams)
        - Lexicon-based sentiment features (VADER)
        - Classical ML models (Logistic Regression, Linear SVM, Naive Bayes)
        """
    )

    st.caption("⚠️ This tool is for educational and demonstration purposes only.")
