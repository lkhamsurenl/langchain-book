import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from document_loader import DocumentLoader
from rag import graph, config, retriever 

st.set_page_config(
    page_title="RAG Agent",
    page_icon=":robot_face:",
    layout="wide",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

for message in st.session_state.chat_history:
    print(f"Message: {message}")
    with st.chat_message("role"):
        st.markdown(message)

docs = retriever.add_uploaded_docs(st.session_state.uploaded_files)

def process_message(message: str):
    response = graph.invoke({
        "messages": HumanMessage(message),
    }, config=config)

    return response["messages"][-1].content

st.markdown("""
# Company Document RAG Agent
""")


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat Interface")

    if user_message := st.chat_input("Enter your message"):
        with st.chat_message("user"):
            st.markdown(user_message)

        st.session_state.chat_history.append({
            "role": "User",
            "content": user_message,
        })
        response = process_message(user_message)

        with st.chat_message("Assistant"):
            st.markdown(response)

        st.session_state.chat_history.append({
            "role": "Assistant",
            "content": response,
        })


with col2:
    st.subheader("Document Management")

    uploaded_files = st.file_uploader("Upload Documents", type=list(DocumentLoader.supported_extensions), accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(file)
