from langchain_anthropic import ChatAnthropic 
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain.agents import load_tools
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)

# High level plan
# 1. research step for the student
research_system_prompt = (
    "You're a hard working student that is trying to do multiple choice questions.\n"
    "Do no assume anything, always use the tools to check your work.\n"
    "You have access to the following tools:\n"
    "* Arxiv search tool\n"
    "* Wikipedia search tool\n"
    "* DuckDuckGo search tool\n"
    "You will be given a question and a set of options.\n"
    "You need to research the question and return the answer.\n"
    "Keep the tool usage to a minimum and only use it if it's absolutely necessary.\n"
)

research_prompt = ChatPromptTemplate.from_messages([
    ("system", research_system_prompt),
    ("user", "Question: {question}\nOptions: {options}\n"),
    ("placeholder", "{messages}")
])
tools = load_tools(["arxiv", "wikipedia", "ddg-search"])

class ResearchState(AgentState):
    """State for the research agent."""
    question: str
    options: list[str]

research_agent = create_react_agent(
    model=llm,
    prompt=research_prompt,
    tools=tools,
    state_schema=ResearchState,
)

# 2. professor critique the response
critique_system_prompt = (
    "You're a professor that is critiquing a student's response to a multiple choice question.\n"
    "You will be given a question and a set of options.\n"
    "You need to critique the student's response and return the critique.\n"
    "Keep the critique to a minimum and only use it if it's absolutely necessary.\n"
    "Return the original response and no critique if the student's response is correct."
)

critique_prompt = ChatPromptTemplate.from_messages([
    ("system", critique_system_prompt),
    ("user", "Question: {question}\nOptions: {options}\nStudent Response: {response}\n"),
    ("placeholder", "{messages}")
])

class CritiqueResponse(BaseModel):
    critique: Optional[str] = Field(description="The critique of the student's response", default=None)
    answer: Optional[str] = Field(description="Student's answer to the question", default=None)

critique_chain = (critique_prompt | llm.with_structured_output(CritiqueResponse))

# 3. student revise the response based on the critique
class ReviseResponse(ResearchState):
    response: Optional[str]
    critique: Optional[str]

revise_research_system_prompt = (
    "You're a student that is revising a response to a multiple choice question.\n"
    "You need to revise the the response based on the critique and return the revised response.\n"
    "Keep the revised response to a minimum and only use tools if it's absolutely necessary.\n"
)
revise_research_prompt = ChatPromptTemplate.from_messages([
    ("system", revise_research_system_prompt),
    ("user", "Question: {question}\nOptions: {options}\nCritique: {critique}\nStudent Response: {response}\n"),
    ("placeholder", "{messages}")
])
revise_research_agent = create_react_agent(
    model=llm,
    prompt=revise_research_prompt,
    tools=tools,
    state_schema=ReviseResponse,
)

class ResearchGraphState(TypedDict):
    question: str 
    options: str
    critique: Optional[str]
    response: Optional[str]

def _research_node(state: ResearchGraphState) -> ResearchGraphState:
    result = research_agent.invoke(state)
    return {
        "question": state["question"], 
        "options": state["options"],
        "response": result["messages"][-1].content
    }

def _critique_node(state: ResearchGraphState) -> ResearchGraphState:
    result: CritiqueResponse = critique_chain.invoke(state)
    return {
        "critique": result.critique, 
        "response": result.answer
    }


def _revise_node(state: ResearchGraphState) -> ResearchGraphState:
    result: ReviseResponse = revise_research_agent.invoke(state)
    return {
        "response": result["messages"][-1].content,
        "critique": state["critique"],
        "question": state["question"],
        "options": state["options"]
    }

def _should_end(state: ResearchGraphState):
    if state["critique"] is not None:
        return "revise"
    return "end"

graph = (
    StateGraph(ResearchGraphState)
    .add_node("research_node", _research_node)
    .add_node("critique_node", _critique_node)
    .add_node("revise_node", _revise_node)
    .add_edge(START, "research_node")
    .add_edge("research_node", "critique_node")
    .add_conditional_edges("critique_node", _should_end, {
        "revise": "revise_node",
        "end": END
    })
    .add_edge("revise_node", "critique_node")
    .compile()
)

for _, event in graph.stream({
    "question": "The main factor preventing subsistence economies from advancing economically is the lack of",
    "options": '1: a currency.\n2: a well-connected transportation infrastructure.\n3: government activity.\n4: a banking service.'
}, stream_mode=["updates"]):
    print(event)