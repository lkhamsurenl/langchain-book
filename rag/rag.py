from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from typing import List, Annotated
from llms import chat_model
from langchain_core.documents import Document
from retriever import DocumentRetriever
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver

system_prompt = (
    "You're a helpful AI assistant. Given a user question "
    "and some company document snippets, write documentation."
    "if none of the documents is relevant document, and then "
    "answer the question to the best of your knowledge."
    "\n\nHere are the documents: "
    "{context}"
)

final_prompt = (
    "Revise the following documentation to be more concise and clear using The Elements of Style\n"
    "Original Document: {answer}"
    "Always return the full revised document, even if not changes are needed."
)


retriever = DocumentRetriever()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", final_prompt)
    ]
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    messages: Annotated[list, add_messages]

def retrieve(state: State):
    retrieved_docs = retriever.invoke(state["messages"][-1].content)
    print(retrieved_docs)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({
        "question": state["messages"][-1].content,
        "context": docs_content
    })
    response = chat_model.invoke(messages)
    print(response.content)
    return {
        "answer": response.content
    }

def doc_finalizer(state: State):
    final_prompt_template = final_prompt.invoke({
        "answer": state["answer"]
    })
    response = chat_model.invoke(final_prompt_template)
    print(f"doc_finalizer: {response}")
    return {
        "messages": [AIMessage(response.content)]
    }

graph_builder = StateGraph(State).add_sequence(
    [retrieve, generate, doc_finalizer]
)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("doc_finalizer", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {
    "configurable": {"thread_id": "abc123"}
}