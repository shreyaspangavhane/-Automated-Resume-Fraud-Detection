import os
import PyPDF2

# Path for PDF resumes
pdf_folder = "resumes/"
# Path for output folder to save parsed resume text files
output_folder = "output/"
os.makedirs(output_folder, exist_ok=True)

# Function to parse a PDF and extract text
def parse_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Loop through PDF files in the folder and parse them
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        file_path = os.path.join(pdf_folder, filename)
        print(f"Parsing {file_path}...")
        resume_text = parse_pdf(file_path)
        
        # Save parsed resume text to .txt file
        txt_file_path = os.path.join(output_folder, f"{filename.replace('.pdf', '.txt')}")
        with open(txt_file_path, 'w', encoding='utf-8') as out_file:
            out_file.write(resume_text)
        
        print(f"Content of {filename} has been saved to {txt_file_path}")
