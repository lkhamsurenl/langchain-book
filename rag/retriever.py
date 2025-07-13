import os
import tempfile
from typing import List, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_document
from llms import EMBEDDINGS

VECTOR_STORE = InMemoryVectorStore(embedding=EMBEDDINGS)

def split_documents(docs: List[Document]) -> List[Document]:
    # Split documents into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200
    )
    return text_splitter.split_documents(docs)

class DocumentRetriever(BaseRetriever):
    documents: List[Document] = []
    k: int = 5

    def model_post_init(self, ctx: Any) -> None:
        self.store_documents(self.documents)

    @staticmethod
    def store_documents(docs: List[Document]) -> None:
        splits = split_documents(docs)
        VECTOR_STORE.add_documents(splits)

    def add_uploaded_docs(self, uploaded_files):
        # Add list of uploaded files to the vector store
        docs = []
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    try:
                        temp_filepath = os.path.join(temp_dir, file.name)
                        with open(temp_filepath, "wb") as f:
                            f.write(file.getvalue())
                        loaded_docs = load_document(temp_filepath)
                        docs.extend(loaded_docs)
                    except (IOError, OSError) as e:
                        print(f"Error processing file {file.name}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error loading document {file.name}: {e}")
                        continue
                
                if docs:
                    self.documents.extend(docs)
                    self.store_documents(docs)
        except Exception as e:
            print(f"Error creating temporary directory: {e}")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        using default similarity search, find top k most relevant documents.
        """
        if len(self.documents) == 0:
            return []
        return VECTOR_STORE.similarity_search(query=query, k=self.k)


