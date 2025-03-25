# Install dependencies (Run this locally before deployment)
# pip install streamlit scikit-learn python-docx PyPDF2 nltk

import streamlit as st
import pickle
import docx
import PyPDF2
import re
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import shutil
import os

# Get NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')

# Delete nltk_data folder (force redownload)
if os.path.exists(nltk_data_path):
    shutil.rmtree(nltk_data_path)

# Make sure the folder exists
os.makedirs(nltk_data_path, exist_ok=True)

# Re-download all required datasets
nltk.data.path.append(nltk_data_path)
nltk.download('punkt_tab')
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('resumepred.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))


# Function to clean resume text
def clean_resume(txt):
    txt = txt.lower()
    txt = re.sub(r'http\S+|www\S+', '', txt)  # Remove URLs
    txt = re.sub(r'@\w+|#\w+', '', txt)  # Remove mentions & hashtags
    txt = re.sub(r'[^a-zA-Z\s]', '', txt)  # Keep only letters & spaces
    txt = re.sub(r'\s+', ' ', txt).strip()  # Remove extra spaces

    tokens = word_tokenize(txt)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


# Function to extract text from TXT
def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')


# Function to predict the category of a resume
def predict_category(input_resume):
    cleaned_text = clean_resume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume (PDF, TXT, DOCX) and get the predicted job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file:
        try:
            ext = uploaded_file.name.split('.')[-1].lower()
            resume_text = (
                extract_text_from_pdf(uploaded_file) if ext == 'pdf' else
                extract_text_from_docx(uploaded_file) if ext == 'docx' else
                extract_text_from_txt(uploaded_file)
            )

            st.success("Text extraction successful!")

            if st.checkbox("Show extracted text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Category")
            st.write(f"**{predict_category(resume_text)}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 10px;
                center: 0px;
                font-size: 14px;
                color: gray;
            }
        </style>
        <div class="footer">
            A Learning Experiment 🚀 by <b>Huzaifa Ahmed Zaidi</b>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
