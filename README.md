# 💬 Threads Sentiment Analysis

An end-to-end **Natural Language Processing (NLP)** project for **3-class sentiment analysis**
(Positive / Neutral / Negative) on **Threads (Instagram app) user reviews**.

The project covers the full NLP workflow — from data exploration and preprocessing to
model training, evaluation, and deployment via an interactive **Streamlit web app**.

🔗 **Live App:** https://threads-sentiment-analysis-78orbyxozpc.streamlit.app/

---

## 📌 Project Overview

- **Task:** 3-class sentiment classification  
- **Domain:** Threads (Instagram app) reviews  
- **Dataset Size:** ~33,000 reviews  
- **Labels:**  
  - ⭐ 1–2 → Negative  
  - ⭐ 3 → Neutral  
  - ⭐ 4–5 → Positive  

The primary goal is not maximizing accuracy at all costs, but building a **well-justified,
academically sound NLP system** and demonstrating understanding of model behavior,
limitations, and deployment considerations.

---

## 🧰 Tech Stack

### Languages & Libraries
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-numerical-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-data-green?logo=pandas)
![SciPy](https://img.shields.io/badge/SciPy-scientific-lightgrey?logo=scipy)
![NLTK](https://img.shields.io/badge/NLTK-NLP-yellow)
![Joblib](https://img.shields.io/badge/Joblib-serialization-red)

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-visualization-lightblue)

### Deployment
![Streamlit](https://img.shields.io/badge/Streamlit-webapp-ff4b4b?logo=streamlit)
![Streamlit Cloud](https://img.shields.io/badge/Streamlit-Cloud-lightgrey)

---

## 🧠 Methodology

### 1️⃣ Data Processing & EDA
- Dataset inspection and cleaning
- Class imbalance analysis
- Review length distribution
- Label engineering from star ratings

### 2️⃣ Text Preprocessing
- Lowercasing
- URL removal
- Non-alphabetic character removal
- Lightweight token normalization  
> Note: Preprocessing was simplified at deployment time for robustness.

### 3️⃣ Feature Engineering
- **TF-IDF Vectorization**
  - Unigrams + bigrams
  - Frequency filtering (`min_df`, `max_df`)
- **VADER Sentiment Scores**
  - `neg`, `neu`, `pos`, `compound`
- Final feature set: **TF-IDF + VADER**

### 4️⃣ Models Trained
- **Logistic Regression**
- **Linear Support Vector Machine (SVM)**
- **Multinomial Naive Bayes**

All models were evaluated using:
- Accuracy
- Precision / Recall / F1-score
- Macro F1 (to handle class imbalance)
- Confusion matrices

---

## 📊 Key Observations

- **Neutral sentiment** is the hardest class to predict due to:
  - Linguistic ambiguity
  - Overlap with weakly positive/negative language
- **Linear SVM** is more conservative and handles borderline neutral cases better
- **Naive Bayes** tends to favor majority classes due to independence assumptions
- Accuracy alone is insufficient — **per-class performance matters**

---

## 🌐 Web Application

The project includes an interactive **Streamlit app** with:

### 🔮 Predict Tab
- Enter a custom Threads review
- Choose between:
  - Linear SVM (recommended)
  - Logistic Regression
  - Naive Bayes
- Color-coded sentiment output
- Confidence score (Logistic Regression only)

### 📊 Model Insights Tab
- Model behavior explanations
- Performance comparison plots

### ℹ️ About & Limitations Tab
- Methodology summary
- Known limitations
- Academic justification

---

## 📁 Project Structure

```markdown

```text
threads_sentiment_project/
│
├── app/
│   └── app.py
│
├── data/
│   └── threads_reviews.csv
│
├── models/
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_regression_combined.joblib
│   ├── linear_svm_combined.joblib
│   └── naive_bayes_combined.joblib
│
├── plots/
│   ├── sentiment_distribution.png
│   ├── review_length_distribution.png
│   ├── review_length_distribution_zoomed.png
│   ├── accuracy_macro_f1_comparison.png
│   └── confusion_matrix_*.png
│
├── notebooks/
│   └── threads_sentiment_analysis.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚠️ Limitations

- Sentiment labels are inferred from star ratings and may contain noise.
- Neutral sentiment is inherently subjective and ambiguous.
- The model relies solely on textual content (no user or contextual metadata).
- This system is intended for educational and demonstrative purposes.

---

## 🎓 Academic Context

This project was developed as part of a **university NLP coursework**.  
Model choices, evaluation metrics, and design decisions were guided by **academic justification**
rather than maximizing raw performance.

---

## 🚀 Future Improvements

- Transformer-based models (e.g., BERT) for contextual understanding
- Aspect-based sentiment analysis
- Multilingual sentiment support
- Model explainability at inference time
