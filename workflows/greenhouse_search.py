from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from pydantic_core.core_schema import str_schema
import requests 
from typing import Dict, List, TypedDict
from datetime import datetime, timezone
from dateutil import parser
from utils import get_resume_data, get_url_content, create_cover_letter

llm = ChatAnthropic(model="claude-3-7-sonnet-latest")
companies: list[str] = [
    "adobe",
    "affirm",
    "airbnb",
    "amazon",
    "anthropic",
    "apple",
    "block",
    "box",
    "coinbase",
    "datadog",
    "discord",
    "doordash",
    "dropbox",
    "gitlab",
    "linkedin",
    "microsoft",
    "netflix",
    "nvidia",
    "openai",
    "pinterest",
    "slack",
    "snowflake",
    "sourcegraph",
    "square",
    "ssi",
    "stripe",
    "thinkingmachines",
    "uber",
    "yelp",
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


def get_active_jobs(company_name: str, role: str = "machine", location: str = "remote") -> List[Job]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{company_name}/jobs"
    output: List[Job] = []
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        jobs = data.get("jobs", [])
        
        for job in jobs:
            try:
                if not is_match(job, role, location):
                    continue
                description = get_url_content(job["absolute_url"])
                output.append(Job(
                    company=company_name,
                    title=job["title"], 
                    location=job["location"]["name"], 
                    url=job["absolute_url"], 
                    description=description)
                )
            except (KeyError, TypeError) as e:
                print(f"Error processing job data for {company_name}: {e}")
                continue
                
    except requests.exceptions.RequestException as e:
        print(f"Error fetching jobs for {company_name}: {e}")
    except ValueError as e:
        print(f"Error parsing JSON response for {company_name}: {e}")

    return output





def main():
    resume_str: str = get_resume_data()
    # 1. find all jobs in greenhouse 
    jobs: List[Job] = []
    for company in companies:
        jobs.extend(get_active_jobs(company))

    print(f"Number of jobs found: {len(jobs)}!")

    for job in jobs:
        print(f"\n---------------{job['company']}; {job['url']}\n-------------")


if __name__ == "__main__":
    main()
