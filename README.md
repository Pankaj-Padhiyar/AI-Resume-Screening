# 🚀 AI-Powered Resume Screening & Candidate Ranking

An AI-based web application that **automatically ranks resumes** based on a given **job description** using **TF-IDF** and **cosine similarity**. Built with **Streamlit**, **scikit-learn**, and **PyPDF2**.

---

## 🔹 Features
- 📂 **Upload multiple resumes (PDF)**
- 📌 **Enter a job description**
- 🎯 **Ranks resumes based on similarity**
- 📊 **Displays ranking in a sorted table**
- 🎨 **Custom UI with background styling**

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **Libraries:** scikit-learn, PyPDF2, pandas

---

## 🚀 Installation & Setup

1️⃣ **Clone the repository**  

git clone https://github.com/yourusername/resume-screening.git
cd resume-screening
2️⃣ (Optional) Create a Virtual Environment

python -m venv .venv

Activate on Windows: source .venv\Scripts\activate

Activate on Mac/Linux: source .venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run the Application

streamlit run app.py

The app will open in your default web browser.

📌 How It Works

Upload PDF resumes using the file uploader.

Enter a job description in the provided text area.

The system extracts text from the resumes and converts them into TF-IDF vectors.

The job description is compared against resumes using cosine similarity.

Ranked results are displayed with similarity scores.

📡 Deployment

Deploy on Streamlit Cloud (Free)

Push your project to GitHub.

Go to Streamlit Cloud and log in.

Click New App, select your GitHub repo, and set the file path to app.py.

Click Deploy 🚀

🔧 Requirements

streamlit PyPDF2 pandas scikit-learn numpy

Install using:

pip install -r requirements.txt

🎯 End Users

HR & Recruiters – Automates resume screening for faster hiring.

Hiring Managers – Ranks resumes efficiently for job roles.

Job Portals – Enhances resume-job matching accuracy.

AI Enthusiasts & Students – Learn NLP-based resume analysis.

🔮 Future Scope

✅ AI-Based Scoring – Use ML/Deep Learning for better ranking.✅ Advanced NLP – Integrate BERT/GPT for deeper analysis.✅ Multi-Format Support – Add DOCX, TXT & OCR for images.✅ Skill Matching – Extract skills & experience automatically.✅ API Integration – Connect with job portals & HR systems.

🔚 Conclusion

This AI-powered Resume Screening & Ranking System addresses the challenge of manual resume screening, which is time-consuming and inefficient. By leveraging TF-IDF and Cosine Similarity, the system automates resume ranking, ensuring fast, objective, and accurate candidate shortlisting. With PDF text extraction, real-time ranking, and easy deployment via Streamlit, this project provides an efficient, scalable, and user-friendly solution for recruiters, hiring managers, and job portals. 🚀
