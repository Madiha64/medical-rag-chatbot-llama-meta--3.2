import re

from flask import Flask, render_template, request
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================================
# HUGGING FACE LOGIN
# ==========================================


# login("YOUR_HF_TOKEN")

app = Flask(__name__)

# ==========================================
# LOAD EMBEDDINGS
# ==========================================

print("Loading embeddings...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings loaded!")

# ==========================================
# LOAD FAISS
# ==========================================

print("Loading FAISS...")

vectorstore = FAISS.load_local(
    "medical_faiss",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

print("FAISS loaded!")

# ==========================================
# LOAD LLM
# ==========================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Model loaded!")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    temperature=0.1
)

print("Pipeline ready!")

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# ASK QUESTION
# ==========================================

@app.route("/ask", methods=["POST"])
def ask():

    question = request.form["question"]

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # DEBUG
    print("\n" + "="*80)
    print("QUESTION:")
    print(question)

    print("\nRETRIEVED CONTEXT:")
    print(context[:3000])

    print("="*80 + "\n")

    
    prompt = f"""
You are a medical assistant.

Use ONLY the information found in the context.

Do not use your own medical knowledge.

If the answer cannot be found in the context, reply exactly:

I could not find the answer in the provided medical documents.


Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt)[0]["generated_text"]

    answer = result.replace(prompt, "").strip()

    answer = answer.split("\n\n")[0].strip()

    answer = re.sub(r"\s+", " ", answer)

    matches = list(re.finditer(r"[.!?]", answer))

    if matches:
        answer = answer[:matches[-1].end()]

    answer = answer.strip()

    sources = []

    for doc in docs[:3]:

        page = doc.metadata.get("page", "Unknown")

        source = doc.metadata.get(
            "source",
            "Public Health in Pharmacy Practice_ A Casebook 2nd Edition.pdf"
        )

        sources.append(
            f"Page {page} | {source}"
        )

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        sources=sources
    )


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)