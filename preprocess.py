import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Folder paths
output_folder = "output/"
processed_folder = "processed_output/"
os.makedirs(processed_folder, exist_ok=True)  # Create processed_output folder

# Function to clean and preprocess resume text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Apply stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    
    # Rejoin tokens to form cleaned text
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Loop through and preprocess each resume text file
for filename in os.listdir(output_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(output_folder, filename)
        
        # Read the text content
        with open(file_path, 'r', encoding='utf-8') as file:
            resume_text = file.read()
        
        # Preprocess the text
        processed_text = preprocess_text(resume_text)
        
        # Save processed text to a new file
        processed_file_path = os.path.join(processed_folder, filename)
        with open(processed_file_path, 'w', encoding='utf-8') as file:
            file.write(processed_text)
        
        print(f"Processed {filename} and saved to {processed_file_path}")
