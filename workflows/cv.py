from utils import get_resume_data, get_url_content, create_cover_letter
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
import argparse
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import Optional, TypedDict

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.0,
    max_tokens=1000
)
class JobDescription(BaseModel):
    """
    State for the job description extraction agent
    """
    extracted_job_description: str = Field(description="Extracted job description content")

jd_system_prompt = (
    "You're a helpful assistant "
    "Given full HTML string, extract relevant "
    "job description part of it."
    "Do not return anything else."
)
job_description_prompt = ChatPromptTemplate.from_messages([
    ("system", jd_system_prompt),
    ("human", "Full HTML string: {job_url_content}")
])
job_description_chain = (job_description_prompt | llm.with_structured_output(JobDescription))


critique_system_prompt = (
    "You're an expert resume reviewer that is critiquing a cover letter.\n"
    "You will be given a job description and cover letter.\n"
    "You need to critique the the cover letter and return the critique.\n"
    "Keep the critique to a minimum and only use it if it's absolutely necessary.\n"
    "Return revised version of the cover letter and no critique if the cover letter quality is satisfactory."
)

critique_prompt = ChatPromptTemplate.from_messages([
    ("system", critique_system_prompt),
    ("user", "Job Description: {job_description}\n\nResume:\n{resume_str}\n\nCover Letter\n:{cover_letter}\n"),
    ("placeholder", "{messages}")
])
class CritiqueResponse(BaseModel):
    critique: Optional[str] = Field(description="Critique to the cover letter")
    cover_letter: str = Field(description="Improved cover letter that addresses critiques")
critique_chain = (critique_prompt | llm.with_structured_output(CritiqueResponse))

class RevisedCoverLetter(BaseModel):
    cover_letter: str = Field(description="Revised cover letter that addressed critique points")
revise_system_prompt = (
    "You're job applicant that is revising your cover letter "
    "given critique from an expert. You need to revise your cover letter "
    "to address the critique points."
)
revise_cv_prompt = ChatPromptTemplate.from_messages([
    ("system", revise_system_prompt),
    ("user", "Job Description: {job_description}\nResume:\n{resume_str}\nCover Letter\n:{cover_letter}\nCritique:{critique}"),
    ("placeholder", "{messages}")
])
revise_chain = (revise_cv_prompt | llm.with_structured_output(RevisedCoverLetter))

class JobCoverLetterState(TypedDict):
    resume_str: str
    job_url_content: str
    job_description: str 
    cover_letter: str 
    critique: Optional[str]

def _job_description_node(state: JobCoverLetterState):
    result: JobDescription = job_description_chain.invoke({
        "job_url_content": state["job_url_content"]
    })
    return {
        "job_description": result.extracted_job_description
    }

def _cover_letter_node(state: JobCoverLetterState):
    result = create_cover_letter(
        llm=llm,
        job_description=state["job_description"],
    )
    return {
        "cover_letter": result
    }

def _cover_letter_critique_node(state: JobCoverLetterState):
    result: CritiqueResponse = critique_chain.invoke({
        "job_description": state["job_description"],
        "resume_str": state["resume_str"],
        "cover_letter": state["cover_letter"]
    })

    return {
        "critique": result.critique,
        "cover_letter": result.cover_letter
    }

def _revise_node(state: JobCoverLetterState):
    result: RevisedCoverLetter = revise_chain.invoke({
        "job_description": state["job_description"],
        "resume_str": state["resume_str"],
        "cover_letter": state["cover_letter"],
        "critique": state["critique"],
    })
    return {
        "cover_letter": result.cover_letter
    }

def _should_revise_cover_letter(state: JobCoverLetterState):
    if state["critique"] is not None:
        return "revise"
    else:
        return "end"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    args = parser.parse_args()

    resume_str: str = get_resume_data()
    job_url_content = get_url_content(args.url)

    graph = (
        StateGraph(JobCoverLetterState)
        .add_node("job_description_node", _job_description_node)
        .add_node("cover_letter_node", _cover_letter_node)
        .add_node("cover_letter_critique_node", _cover_letter_critique_node)
        .add_node("revise_node", _revise_node)
        .add_edge(START, "job_description_node")
        .add_edge("job_description_node", "cover_letter_node")
        .add_edge("cover_letter_node", "cover_letter_critique_node")
        .add_conditional_edges("cover_letter_critique_node", _should_revise_cover_letter, {
            "revise": "revise_node",
            "end": END
        })
        .compile()
    )

    result = graph.invoke({
        "resume_str": resume_str,
        "job_url_content": job_url_content
    })
    print(result["cover_letter"])


if __name__ == "__main__":
    main()