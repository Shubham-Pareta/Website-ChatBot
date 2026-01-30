from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import os

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 🔹 Build vector database
def process_website(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    docs = splitter.create_documents(texts, metadatas=metadatas)
    return Chroma.from_documents(docs, embedding_model)


# 🔹 Question answering
def get_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    # ⚠️ Small context to avoid Groq rejection
    context = "\n".join(d.page_content[:200] for d in docs)

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY from the website context. "
                           "If the answer is not present, reply exactly: "
                           "The answer is not available on the provided website."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        temperature=0,
        max_tokens=200
    )

    return response.choices[0].message.content
