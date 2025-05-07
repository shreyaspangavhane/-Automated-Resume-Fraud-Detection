# rule_based_check.py

import re

# Updated suspicious keywords
suspicious_keywords = [
    "ninja", "guru", "rockstar", "wizard", "superstar",
    "master of all", "top-tier", "tech evangelist",
    "coding god", "10+ years of experience",
    "20 years industry knowledge",
    "expert in all technologies",
    "handled hundreds of clients",
    "worked on 100+ projects"
]

# Fake universities can stay empty for now or be customized
fake_universities = ["University of Nowhere", "Fake State University"]

def rule_based_fraud_check(resume_text):
    warnings = []

    # Check for suspicious keywords
    for keyword in suspicious_keywords:
        if keyword.lower() in resume_text.lower():
            warnings.append(f"⚠️ Suspicious keyword: {keyword}")

    return warnings
