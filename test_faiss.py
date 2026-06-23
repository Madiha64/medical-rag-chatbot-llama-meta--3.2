from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_fn = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "medical_faiss",
    embedding_fn,
    allow_dangerous_deserialization=True
)

print("FAISS loaded successfully!")

docs = vectorstore.similarity_search(
    "What is HIV?",
    k=3
)

print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content[:300])
    print("-" * 50)