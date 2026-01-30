from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def process_website(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    docs = splitter.create_documents(texts, metadatas=metadatas)
    return Chroma.from_documents(docs, embedding_model)


def get_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    # Reduce size to avoid token overflow
    context = "\n\n".join(d.page_content[:300] for d in docs)

    llm = ChatGroq(
        model="llama3-70b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    messages = [
        (
            "system",
            "Answer using ONLY the provided website context. "
            "If the answer is not present, reply exactly: "
            "'The answer is not available on the provided website.'"
        ),
        (
            "human",
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )
    ]

    response = llm.invoke(messages)
    return response.content
