import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import defaultdict
import os
from docx import Document
import PyPDF2
import shutil
import zipfile

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Functions from the previous script
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages])

def process_resume_directory(directory_path):
    resume_texts = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if filename.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            elif filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            else:
                continue  # Skip files that are neither DOCX nor PDF
            resume_texts[filename] = text
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
    return resume_texts

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def classify_resume(text):
    classification = defaultdict(str)
    
    # Job role/industry
    job_roles = {
        "software": ["software engineer", "developer", "programmer"],
        "data": ["data scientist", "data analyst", "machine learning"],
        "marketing": ["marketing", "seo", "social media"],
        "finance": ["accountant", "financial analyst", "bookkeeper"],
    }
    
    for role, keywords in job_roles.items():
        if any(keyword in text.lower() for keyword in keywords):
            classification["job role"] = role
            break
    
    # Education level
    education_levels = ["High School", "Associate", "Bachelor", "Master", "PhD"]
    for level in education_levels:
        if level.lower() in text.lower():
            classification["education"] = level
            break
    
    # Years of experience
    experience_match = re.search(r"(\d+)\s*(?:years?|yrs?)(?:\s+of)?\s+experience", text, re.IGNORECASE)
    if experience_match:
        classification["years_experience"] = experience_match.group(1)
    
    # Skills
    skills = ["Python", "Java", "C++", "JavaScript", "SQL", "AWS", "Docker", "Kubernetes",
              "Machine Learning", "Data Analysis", "Project Management", "Agile", "Scrum"]
    found_skills = [skill for skill in skills if skill.lower() in text.lower()]
    classification["skills"] = ", ".join(found_skills)

    # Phone number
    phone_pattern = r'\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        classification["phone number"] = phone_match.group()

    # Address (basic pattern, might need refinement)
    address_pattern = r'\d{1,5}\s\w+\s\w+\.?(?:\s\w+\.?)?\s*,?\s*\w+\s*,?\s*[A-Z]{2}\s*\d{5}'
    address_match = re.search(address_pattern, text)
    if address_match:
        classification["address"] = address_match.group()

    return classification

def create_resume_ranking_model(job_description, resume_directory):
    # Process resumes
    resume_texts = process_resume_directory(resume_directory)
    
    # Classify resumes
    classified_resumes = {filename: classify_resume(text) for filename, text in resume_texts.items()}
    
    # Create DataFrame from classified resumes
    df = pd.DataFrame.from_dict(classified_resumes, orient='index')
    df['filename'] = df.index
    df.reset_index(drop=True, inplace=True)
    
    # Combine relevant columns into a single text field
    df['combined_text'] = df[['education', 'job role', 'skills']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    
    # Add years of experience to the combined text
    df['combined_text'] += ' ' + df['years_experience'].astype(str) + ' years experience'
    
    # Preprocess job description and resumes
    preprocessed_jd = preprocess_text(job_description)
    preprocessed_resumes = df['combined_text'].apply(preprocess_text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the job description and resumes
    tfidf_matrix = vectorizer.fit_transform([preprocessed_jd] + list(preprocessed_resumes))
    
    # Calculate cosine similarity between job description and each resume
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Add similarity scores to the dataframe
    df['similarity_score'] = cosine_similarities
    
    # Sort resumes by similarity score in descending order
    ranked_resumes = df.sort_values('similarity_score', ascending=False).reset_index(drop=True)
    
    return ranked_resumes


# Streamlit app
st.title('Resume Ranking System')

st.write("""
This app ranks resumes based on their similarity to a given job description.
Upload a directory containing resumes (PDF and DOCX formats) and enter a job description to get started.
""")

# Job description input
job_description = st.text_area("Enter the job description:", height=200)

# Resume directory input
resume_directory = st.text_input("Enter the path to the resume directory:")

if st.button('Rank Resumes'):
    if job_description and resume_directory:
        try:
            with st.spinner('Processing resumes...'):
                ranked_resumes = create_resume_ranking_model(job_description, resume_directory)
            
            st.success('Resumes ranked successfully!')
            
            # Display results
            st.write("Top 5 Ranked Resumes:")
            st.dataframe(ranked_resumes.head())
            
            # Create a folder with ranked resumes
            output_folder = "ranked_resumes"
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder)
            
            for index, row in ranked_resumes.iterrows():
                src_file = os.path.join(resume_directory, row['filename'])
                dst_file = os.path.join(output_folder, f"{index+1:03d}_{row['filename']}")
                shutil.copy2(src_file, dst_file)
            
            # Create a zip file of the ranked resumes
            shutil.make_archive(output_folder, 'zip', output_folder)
            
            # Offer the zip file for download
            with open(f"{output_folder}.zip", "rb") as file:
                st.download_button(
                    label="Download ranked resumes as ZIP",
                    data=file,
                    file_name="ranked_resumes.zip",
                    mime="application/zip"
                )
            
            # Option to download full results as CSV
            csv = ranked_resumes.to_csv(index=False)
            st.download_button(
                label="Download full results as CSV",
                data=csv,
                file_name="ranked_resumes.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter both a job description and a resume directory path.")

st.write("""
### How to use this app:
1. Enter the job description in the text area above.
2. Provide the full path to the directory containing the resumes.
3. Click the 'Rank Resumes' button.
4. View the top 5 ranked resumes in the table.
5. Download the ranked resumes as a ZIP file.
6. Download the full results as a CSV file if needed.
""")