from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

app = Flask(__name__)
CORS(app)

# Load environment variables (set your OpenAI API key as an env variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables.")

# Load specific Markdown documents
def load_documents():
    doc_folder = "docs"
    filenames = [
        "NovaTech_Benefits_and_Perks.md",
        "NovaTech_Employee_Terms_and_Conditions.md",
        "NovaTech_Code_of_Conduct.md",
        "NovaTech_PTO_Policy.md",
        "NovaTech_2025_Holiday_List.md",
        "NovaTech_WiFi_and_Reset_Policy.md",
        "NovaTech_IT_Policies.md"
    ]
    docs = []
    for name in filenames:
        full_path = os.path.join(doc_folder, name)
        loader = TextLoader(full_path, encoding="utf8")
        docs.extend(loader.load())
    return docs

# Split and embed documents
docs = load_documents()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Setup FAISS vector store
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(splits, embedding)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Setup QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=retriever
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "")
    if not query:
        return jsonify({"answer": "Please enter a question."})

    try:
        result = qa.run(query)
        return jsonify({"answer": result})
    except Exception as e:
        return jsonify({"answer": f"⚠️ DocDragon error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
