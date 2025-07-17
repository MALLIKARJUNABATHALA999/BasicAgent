from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load dataset
df = pd.read_csv("./realistic_restaurant_reviews.csv")

# Create embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define Chroma DB path
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Prepare documents and IDs if DB doesn't exist
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=f"{row['Title']} {row['Review']}",
            metadata={"rating": row["Rating"], "date": row["Date"]},
        )
        documents.append(document)
        ids.append(str(i))

    # Initialize vector store and add documents
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )
    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()  # Save to disk

else:
    # Load existing vector store
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
