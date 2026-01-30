from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# QA model (extracts answers instead of generating)
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def process_website(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    docs = splitter.create_documents(texts, metadatas=metadatas)
    return Chroma.from_documents(docs, embedding_model)


def get_answer(question, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    best_answer = ""
    best_score = 0

    for d in docs:
        result = qa_pipeline(question=question, context=d.page_content[:1000])

        if result["score"] > best_score and result["answer"].strip():
            best_score = result["score"]
            best_answer = result["answer"]

    if best_score < 0.2:
        return "The answer is not available on the provided website."

    return best_answer
