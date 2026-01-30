from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load local text-generation model (no API needed)
generator = pipeline("text2text-generation", model="google/flan-t5-base")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def process_website(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    docs = splitter.create_documents(texts, metadatas=metadatas)
    return Chroma.from_documents(docs, embedding_model)


def get_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = "\n".join(d.page_content[:250] for d in docs)

    prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question:
{question}

If not found, say:
The answer is not available on the provided website.
"""

    response = generator(prompt, max_length=200)[0]["generated_text"]
    return response
