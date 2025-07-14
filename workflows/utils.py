import base64
import requests
from langchain_anthropic import ChatAnthropic

DEFAULT_RESUME_FILENAME: str = "/Users/lkhamsurenl/development/resume/20250514_Luvsandondov_Lkhamsuren.pdf"

def get_resume_data(resume_filename: str = DEFAULT_RESUME_FILENAME) -> str:
    with open(resume_filename, "rb") as f:
        resume_data = base64.b64encode(f.read()).decode("utf-8")
    return resume_data

def get_url_content(job_url: str) -> str:
    try:
        response = requests.get(job_url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching job description from {job_url}: {e}")
        return ""

def create_cover_letter(llm, job_description: str, resume_filename: str = DEFAULT_RESUME_FILENAME) -> str:
    resume_str: str = get_resume_data(resume_filename=resume_filename)
    system_prompt = f"""
    You are an expert cover letter writer
    You are given a job description and a resume.
    You need to create a cover letter for the job.
    The cover letter should be in the following format:
    
    <cover letter>
    <signature>
    <name>
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Resume:\n",
                },
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": resume_str,
                    "mime_type": "application/pdf",
                    "filename": resume_filename,
                },
                {
                    "type": "text",
                    "text": f"Job Description:\n{job_description}\nCover Letter:"
                }
            ],
        }
    ]

    return llm.invoke(messages).content