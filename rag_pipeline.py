from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatGroq
import os

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def process_website(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    docs = splitter.create_documents(texts, metadatas=metadatas)
    return Chroma.from_documents(docs, embedding_model)


def get_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(d.page_content for d in docs)
    context = context[:1800]

    prompt = f"""
You are a website question-answering assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Rules:
- Do not use outside knowledge
- If answer is missing, reply exactly:
The answer is not available on the provided website.
"""

    llm = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    response = llm.invoke(prompt)
    return response.content
