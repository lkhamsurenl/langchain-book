from langchain_community.llms import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate


def fake_llm() -> None:
    # how to use fake LLM:
    fake_llm = FakeListLLM(responses=["Hello"])
    result = fake_llm.invoke("Any input will return Hello!")
    print(result)


def anthropic() -> None:
    llm = ChatAnthropic(model="claude-3-opus")
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content="Write a python function to create LLM"),
    ]

    response = llm.invoke(messages)
    print(response)


def chain_example() -> None:
    template = PromptTemplate.from_messages([
        ("system", "You're an experienced programmer and mathematical analyst"),
        ("user", "{problem}")
    ])

    chat = ChatAnthropic(
        model="claude-3-7-sonnet",
        max_tokens=64_000,
        thinking={
            "type": "enabled",
            "budget_tokens": 15_000
        }
    )

    chain = template | chat
    problem = "Design an algorithm to create LLM like Claude"
    response = chain.invoke([HumanMessage(content=problem)])
    print(response)


def chain_example2() -> None:
    prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
    llm = ChatOpenAI(model="gpt-4o")
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    result = chain.invoke({"topic": "Hawaii"})
    print(result)

def main():
    chain_example2()


if __name__ == "__main__":
    main()