import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Folder paths
processed_folder = "processed_output/"

# Read all processed resume text files
processed_resumes = []
resume_files = [os.path.join(processed_folder, f) for f in os.listdir(processed_folder) if f.endswith('.txt')]

# Load the text data
for resume_file in resume_files:
    with open(resume_file, 'r', encoding='utf-8') as file:
        processed_resumes.append(file.read())

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Transform the text data into numerical features (TF-IDF matrix)
tfidf_matrix = vectorizer.fit_transform(processed_resumes)

# Convert the matrix into a DataFrame for better visualization (optional)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Print the feature matrix (TF-IDF values) for the first resume as an example
print("TF-IDF feature matrix for the first resume:")
print(tfidf_df.head())  # Shows the top 5 rows of the feature matrix
