from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=None,
    max_retries=2,
)

store = LocalFileStore("./cache/")
# This is a function to generate embeddings, given document
underlying_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)

