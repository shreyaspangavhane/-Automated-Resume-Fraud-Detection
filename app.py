import streamlit as st
import os
import pickle
from parse_resumes import parse_pdf
from preprocess import preprocess_text
from rule_based_check import rule_based_fraud_check

# Load ML model and vectorizer
with open('final_naive_bayes.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('final_tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

st.title("📄 Automated Resume Fraud Detection")

uploaded_file = st.file_uploader("Upload a Resume (PDF only)", type=['pdf'])

if uploaded_file is not None:
    # Save uploaded PDF
    with open(f"temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Parse and preprocess
    resume_text = parse_pdf("temp_resume.pdf")
    processed_text = preprocess_text(resume_text)

    # ML Prediction with probability
    vector = vectorizer.transform([processed_text])
    prob = model.predict_proba(vector)[0]  # Get probabilities
    prediction = model.predict(vector)[0]  # Get final prediction
    label = "Fraudulent" if prediction == 1 else "Legitimate"

    # Display the probabilities and final prediction
    st.subheader("🧠 Machine Learning Prediction:")
    st.write(f"Probability of Legitimate: {prob[0]:.2f}")
    st.write(f"Probability of Fraudulent: {prob[1]:.2f}")
    st.success(f"This resume is predicted to be: **{label}**")

    # Rule-based check
    st.subheader("🛡️ Rule-Based Warnings:")
    warnings = rule_based_fraud_check(resume_text)
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.info("No issues found by rule-based checks.")

    # Show resume text (optional)
    with st.expander("📝 View Extracted Resume Text"):
        st.text(resume_text)
