import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Custom CSS for styling with a background image
st.markdown(
    """
    <style>
    /* Background Image with Overlay */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)),
                    url("https://media.istockphoto.com/id/1149054436/photo/business-man-review-his-resume-application-on-desk-laptop-computer-job-seeker.jpg?s=612x612&w=0&k=20&c=2M_xMNkuEZkg8-zy9dzP16VX8tHRbmghJtE3g6zPR5g=") 
                    no-repeat center fixed;
        background-size: cover;
    }

    /* Main Content Container */
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.2);
        margin: 15px;
    }

    /* Header Box */
    .header-box {
        background-color: #004AAD;
        color: white;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        text-transform: uppercase;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
    }

    /* Upload Box */
    .upload-box {
        background-color: #FFC107;
        color: black;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
    }

    /* Results Box */
    .result-box {
        background-color: #28A745;
        color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
    }

    /* DataFrame Styling */
    table {
        border-collapse: collapse;
        width: 100%;
        background-color: white;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
        border-radius: 5px;
        overflow: hidden;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
        font-size: 16px;
    }
    th {
        background-color: #007BFF;
        color: white;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    tr:hover {
        background-color: #e2e6ea;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .header-box, .upload-box, .result-box {
            font-size: 20px;
            padding: 14px;
        }
        th, td {
            padding: 10px;
            font-size: 14px;
        }
    }
</style>""",
    unsafe_allow_html=True
)

# Display a high-quality banner image with new use_container_width parameter
st.image(
    "https://media.istockphoto.com/id/1909218991/vector/cartoon-color-human-hands-holding-cv-profile-vector.jpg?s=2048x2048&w=is&k=20&c=uLEnVEbLXIJYpbzuby03co9E6P1KyOh2MS-HiTdMIJk=" , 
    width=300,use_container_width=True
)

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Handle NoneType for empty pages
    return text

# Function to rank resumes based on similarity to job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(job_description_vector.reshape(1, -1), resume_vectors).flatten()
    return cosine_similarities

# Streamlit App UI
st.markdown('<div class="header-box">🚀 AI RESUME SCREENING & CANDIDATE RANKING</div>', unsafe_allow_html=True)

st.subheader("📌 JOB DESCRIPTION")
job_description = st.text_area("ENTER THE JOB DESCRIPTION", height=200)

st.markdown('<div class="upload-box">📂 UPLOAD RESUME</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Process uploaded files and rank resumes
if uploaded_files and job_description:
    st.markdown('<div class="result-box">🏆 Ranking Resumes</div>', unsafe_allow_html=True)

    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)

    # Create and display ranking results
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    st.write(results)
