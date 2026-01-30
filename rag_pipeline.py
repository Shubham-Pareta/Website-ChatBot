from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import os

# Local embedding model (no API cost)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Groq client using OpenAI-compatible SDK
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def process_website(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    docs = splitter.create_documents(texts, metadatas=metadatas)

    # Persist DB so it survives reruns
    vectordb = Chroma.from_documents(
        docs,
        embedding_model,
        persist_directory="chroma_store"
    )
    return vectordb


def get_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    if len(question.split()) <= 2:
        question = "Explain " + question

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content[:300] for d in docs)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # ✅ VALID GROQ MODEL
            messages=[
                {
                    "role": "system",
                    "content": "Answer ONLY using the website context. "
                               "If not present, say exactly: "
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

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("LLM ERROR:", e)  # shows in Streamlit logs
        return "⚠️ LLM request failed. Check API key or model name."
