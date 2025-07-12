from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from pydantic_core.core_schema import str_schema
import requests 
from typing import Dict, List, TypedDict
import base64
from datetime import datetime, timezone
from dateutil import parser
import argparse

llm = ChatAnthropic(model="claude-3-7-sonnet-latest")

companies: list[str] = [
    "stripe",
    "discord",
    "anthropic",
    "uber",
    "yelp",
    "datadog",
    "gitlab",
    "netflix",
    "sourcegraph",
    "square",
    "block",
    "box",
    "dropbox",
    "apple",
    "amazon",
    "adobe",
    "microsoft",
    "ssi",
    "openai",
    "thinkingmachines",
    "doordash",
    "airbnb",
    "slack",
    "snowflake",
    "pinterest",
    "nvidia",
    "linkedin",
    "coinbase",
]

class Job(TypedDict):
    company: str
    title: str 
    location: str 
    url: str 
    description: str


def is_within_last_week(date_string):
    """
    Check if a given date string is within the last 7 days.
    
    Args:
        date_string (str): Date string in ISO format (e.g., "2025-05-20T16:49:37-04:00")
    
    Returns:
        bool: True if the date is within the last week, False otherwise
    """
    try:
        # Parse the date string (handles various formats and timezones)
        given_date = parser.parse(date_string)
        
        # Get current time in UTC
        now = datetime.now(timezone.utc)
        
        # Convert given_date to UTC if it has timezone info
        if given_date.tzinfo is not None:
            given_date = given_date.astimezone(timezone.utc)
        else:
            # If no timezone info, assume it's in local timezone
            given_date = given_date.replace(tzinfo=timezone.utc)
        
        # Calculate the difference
        time_diff = now - given_date
        
        # Check if within last 7 days
        return 0 <= time_diff.days <= 7 and time_diff.total_seconds() >= 0
        
    except Exception as e:
        print(f"Error parsing date: {e}")
        return False

def is_match(job: Dict, role: str, location: str) -> bool:
    if role.lower() not in job["title"].lower():
        return False
    if location.lower() not in job["location"]["name"].lower():
        return False
    # ensure the job was posted within last week
    if not is_within_last_week(job["first_published"]):
        return False

    return True

def parse_job_description(job_url: str) -> str:
    response = requests.get(job_url)
    if response.status_code == 200:
        return response.text
    return ""

def get_active_jobs(company_name: str, role: str = "machine", location: str = "remote") -> List[Job]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{company_name}/jobs"
    response = requests.get(url)
    output: List[Job] = []
    if response.status_code == 200:
        data = response.json()
        jobs = data.get("jobs", [])
        for job in jobs:
            if not is_match(job, role, location):
                continue
            description = parse_job_description(job["absolute_url"])
            output.append(Job(
                company=company_name,
                title=job["title"], 
                location=job["location"]["name"], 
                url=job["absolute_url"], 
                description=description)
            )

    return output



def create_cover_letter(job: Job, resume_data: str) -> str:
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
                    "data": resume_data,
                    "mime_type": "application/pdf",
                },
                {
                    "type": "text",
                    "text": f"Job Description:\n{job['description']}\nCover Letter:"
                }
            ],
        }
    ]

    return llm.invoke(messages).content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, required=True)
    args = parser.parse_args()
    resume_path = args.resume
    with open(resume_path, "rb") as f:
        resume_data = base64.b64encode(f.read()).decode("utf-8")

    # 1. find all jobs in greenhouse 
    jobs: List[Job] = []
    for company in companies:
        jobs.extend(get_active_jobs(company))
        if len(jobs) > 0:
            break

    # 2. parse given job description using its link
    for job in jobs:
        cover_letter = create_cover_letter(job, resume_data)
        print(f"Cover letter for {job['title']} at {job['company']}:\n{cover_letter}\n----------------\n")


if __name__ == "__main__":
    main()
