from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
import argparse

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [PythonREPLTool()]

class CodeAgentState(AgentState):
    question: str 

system_prompt = (
    "You're a helpful assistant that can answer questions and execute code."
    "You have access to a Python REPL. You can use it to execute code and get the results."
    "You can also use it to answer questions."
    "You can also use it to execute code and get the results."
    "Do not assume anything."
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{question}"),
        ("placeholder", "{messages}")
    ])

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        state_schema=CodeAgentState,
        debug=True,
    )

    result = agent.invoke({
        "question": args.question,
    })
    print(result['messages'][-1].content)

if __name__ == "__main__":
    main()