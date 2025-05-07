import streamlit as st
import pickle
import os
from parse_resumes import parse_pdf
from preprocess import preprocess_text
from rule_based_check import rule_based_fraud_check

# Load the FINAL trained model and vectorizer
with open('final_naive_bayes.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('final_tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

st.title("📄 Updated Resume Fraud Detection System")

uploaded_file = st.file_uploader("Upload a Resume (PDF only)", type=['pdf'])

if uploaded_file is not None:
    # Save uploaded PDF temporarily
    with open(f"temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Parse the PDF
    resume_text = parse_pdf("temp_resume.pdf")

    # Preprocess
    processed_text = preprocess_text(resume_text)

    # Transform to vector
    vector = vectorizer.transform([processed_text])

    # Predict
    prediction = model.predict(vector)[0]
    label = "Fraudulent" if prediction == 1 else "Legitimate"

    st.subheader("🧠 Machine Learning Prediction:")
    st.success(f"This resume is predicted to be: **{label}**")

    # Rule-Based Analysis
    st.subheader("🛡️ Rule-Based Warnings:")
    warnings = rule_based_fraud_check(resume_text)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.info("No issues found by rule-based checks.")

    # View raw text (optional)
    with st.expander("📝 View Extracted Resume Text"):
        st.text(resume_text)
