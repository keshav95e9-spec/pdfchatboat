from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Load LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():

    global vectorstore

    file = request.files["pdf"]

    filename = secure_filename(file.filename)

    filepath = os.path.join("Docs", filename)

    file.save(filepath)

    # Load PDF
    loader = PyPDFLoader(filepath)

    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    # Store vectors
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./Vector"
    )

    return "PDF Uploaded Successfully"


@app.route("/ask", methods=["POST"])
def ask_question():

    question = request.form["question"]

    docs = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    response = llm.invoke(f"""
    Answer using this context:

    {context}

    Question:
    {question}
    """)

    return response.content


if __name__ == "__main__":
    app.run(debug=True)