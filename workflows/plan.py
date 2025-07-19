from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.agent_toolkits.load_tools import load_tools
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END
import asyncio

llm = ChatOpenAI(model="gpt-4o-mini")

class Plan(BaseModel):
    """A plan to solve the task"""
    steps: list[str] = Field(description="List of steps necessary to solve the task, should be in sorted order")

system_prompt = (
    "For the given task, come up with a step by step plan.\n"
    "This plan should involve individual tasks, that if executed correctly will "
    "yield the correct answer. Do not add any superfluous steps.\n"
    "The result of the final step should be the final answer. Make sure that each "
    "step has all the information needed - do not skip steps."
)
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "Prepare a plan how to solve the following task:\n{task}\n"),
    ("placeholder", "{messages}")
])

planner = (planner_prompt | llm.with_structured_output(Plan))

executor_system_prompt = (
    "You are given a specific step in full plan to solve a task.\n"
    "Provide answer to the step, using tools when appropriate\n"
    "Do not assume anything.\n"
)
executor_prompt = ChatPromptTemplate.from_messages([
    ("system", executor_system_prompt),
    ("user", "TASK:\n{task}\nPLAN:{plan}\nSTEP:{step}"),
    ("placeholder", "{messages}")
])
class ExecutorState(AgentState):
    task: str 
    plan: Plan
    step: str

tools = load_tools(
    tool_names=["ddg-search", "arxiv", "wikipedia"],
    llm=llm
)
executor_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=executor_prompt,
    state_schema=ExecutorState
)

class PlanState(TypedDict):
    task: str 
    plan: Plan 
    past_steps: Annotated[list[str], operator.add]
    final_response: str 

def get_current_step(state: PlanState) -> int:
    """Returns the current step in the plan"""
    return len(state.get("past_steps", []))

def get_full_plan(state: PlanState):
    """returns formatted plan with step numbers and past results"""
    full_plan = []
    for i, step in enumerate(state["plan"].steps):
        full_step = f"#{i+1}. Planned step: {step}\n"
        if i < get_current_step(state):
            full_step += f"Result: {state['past_steps'][i]}\n"
        full_plan.append(full_step)

    return "\n".join(full_plan)

final_prompt = PromptTemplate.from_template(
    "You're helpful assistant that has executed on a plan."
    "Given teh results of teh execution, prepare the final response.\n"
    "Do not assume anything\nTASK:\n{task}\n\nPLAN WITH RESULTS:\n{plan}\n"
    "FINAL RESPONSE:\n"
)

async def _build_initial_plan(state: PlanState):
    result = await planner.ainvoke({
        "task": state["task"]
    })
    print(f"Plan: {result}")
    return {
        "plan": result
    }

async def _run_step(state: PlanState):
    plan = state["plan"]
    current_step: int = get_current_step(state)
    result = await executor_agent.ainvoke({
        "task": state["task"],
        "plan": get_full_plan(state),
        "step": plan.steps[current_step]
    })
    output = result["messages"][-1].content
    print(f"Output of execution: {output}")
    return {
        # add output from react execution agent to the past steps list
        "past_steps": [output]
    }

async def _get_final_respons(state: PlanState): 
    result = await (final_prompt | llm).ainvoke({
        "task": state["task"],
        "plan": get_full_plan(state)
    })
    print(f"final response: {result.content}")
    return {
        "final_response": result.content
    }

def _should_continue(state: PlanState) -> Literal["run", "final_response"]:
    if get_current_step(state) < len(state["plan"].steps):
        return "run"
    else:
        return "final_response"
    
async def main():
    graph = (
        StateGraph(PlanState)
        .add_node("build_initial_plan", _build_initial_plan)
        .add_node("run_step", _run_step)
        .add_node("get_final_respons", _get_final_respons)
        .add_edge(START, "build_initial_plan")
        .add_edge("build_initial_plan", "run_step")
        .add_conditional_edges(
            source="run_step", 
            path=_should_continue, 
            path_map={
                "run": "run_step",
                "final_response": "get_final_respons",
            }
        )
        .add_edge("get_final_respons", END)
        .compile()
    )

    result = await graph.ainvoke({
        "task": "Write a strategic one-pager of building an AI startup"
    })
    print(result)

if __name__ == "__main__":
    asyncio.run(main())