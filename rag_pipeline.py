from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 🔹 Build Vector DB
def process_website(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    docs = splitter.create_documents(texts, metadatas=metadatas)
    vectordb = Chroma.from_documents(docs, embedding_model)

    return vectordb


# 🔹 Answer questions
def get_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # ✅ NEW LangChain syntax
    docs = retriever.invoke(question)

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
