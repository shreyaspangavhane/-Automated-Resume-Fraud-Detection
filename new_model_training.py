import os
import pickle
import PyPDF2
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Folder where resumes are stored
resume_folder = "resumes/"  # Change if needed

# List of files and their labels
resume_files = [
    "resume1.pdf",
    "resume2.pdf",
    "resume3.pdf",
    "resume4.pdf",
    "resume5.pdf",
    "resume7.pdf",
    "resume8.pdf",
    "resume9.pdf",
    "resume10.pdf",
    "RahulPatilResume (1).pdf"
]
labels = [
    0,  # resume1.pdf
    0,  # resume2.pdf
    0,  # resume3.pdf
    0,  # resume4.pdf
    0,  # resume5.pdf
    1,  # resume7.pdf (Fake)
    0,  # resume8.pdf
    1,  # resume9.pdf (Fake)
    1,  # resume10.pdf (Fake)
    0   # RahulPatilResume (Real)
]

# Function to parse a PDF file and extract text
def parse_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load and preprocess resumes
processed_resumes = []
for filename in resume_files:
    full_path = os.path.join(resume_folder, filename)
    text = parse_pdf(full_path)
    processed_text = preprocess_text(text)
    processed_resumes.append(processed_text)

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_resumes)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open('final_naive_bayes.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('final_tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\n✅ Model and vectorizer have been trained and saved.")
