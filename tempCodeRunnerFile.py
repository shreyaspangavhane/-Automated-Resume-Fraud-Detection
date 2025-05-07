import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Folder path to preprocessed resumes
processed_folder = "processed_output/"

# Read all processed resume text files
processed_resumes = []
resume_files = [os.path.join(processed_folder, f) for f in os.listdir(processed_folder) if f.endswith('.txt')]

# Load the text data
for resume_file in resume_files:
    with open(resume_file, 'r', encoding='utf-8') as file:
        processed_resumes.append(file.read())

# Example labels for the resumes (0 = legitimate, 1 = fraudulent)
# In real scenarios, replace this with actual labels from your dataset
labels = [0, 1]  # Assuming we have two resumes, one legitimate and one fraudulent

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Transform the text data into numerical features (TF-IDF matrix)
tfidf_matrix = vectorizer.fit_transform(processed_resumes)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

# Initialize the Naïve Bayes classifier
nb_classifier = MultinomialNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer for future use
with open('naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump(nb_classifier, model_file)
    
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved.")
