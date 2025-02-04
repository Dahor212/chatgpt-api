import os
import openai
import chromadb
import psutil
from docx import Document
import tiktoken
from flask import Flask, request, jsonify
from flask_cors import CORS

# Nastavení OpenAI API klíče
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializace ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "dokumenty_kolekce"
collection = client.get_or_create_collection(name=collection_name)

app = Flask(__name__)
CORS(app)

# Endpoint pro sledování využití paměti
@app.route("/api/memory", methods=["GET"])
def memory_usage():
    mem = psutil.virtual_memory()
    return jsonify({
        "total": mem.total / 1024**2,
        "used": mem.used / 1024**2,
        "available": mem.available / 1024**2,
        "percent": mem.percent
    })

# Generátor pro načítání dokumentů po jednom
def load_documents_from_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".docx"):
            doc_path = os.path.join(directory_path, filename)
            document = Document(doc_path)
            content = "\n".join(para.text for para in document.paragraphs if para.text.strip())
            yield filename, content

# Rozdělení textu na menší části
def split_text(text, max_tokens=500):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield enc.decode(tokens[i:i + max_tokens])

# Vytvoření embeddingů a uložení do ChromaDB
def create_embeddings(directory_path):
    # Zpracování dokumentů po částech, čímž se šetří paměť
    for doc_name, content in load_documents_from_directory(directory_path):
        for i, chunk in enumerate(split_text(content, max_tokens=500)):
            response = openai.embeddings.create(input=[chunk], model="text-embedding-ada-002")
            embedding = response.data[0].embedding
            collection.add(
                ids=[f"{doc_name}_{i}"],
                embeddings=[embedding],
                metadatas=[{"source": doc_name}],
                documents=[chunk]
            )

# Vyhledání relevantních dokumentů v ChromaDB
def query_chromadb(query, n_results=5):
    response = openai.embeddings.create(input=[query], model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=["documents"])
    return results.get("documents", [])

# Generování odpovědi
def generate_answer_with_assistant(query, context_documents):
    context = "\n\n".join(str(doc) for doc in context_documents)
    prompt = f"""
    Jsi asistentka pro helpdesk ve společnosti, která nabízí penzijní spoření. Tvoje odpovědi musí být založeny pouze na následujících informacích z dokumentů. Pokud žádná odpověď není v těchto dokumentech, odpověz: 'Bohužel, odpověď ve své databázi nemám.'
    
    Kontext dokumentů:
    {context}
    
    Otázka: {query}
    Odpověď:
    """
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Jsi asistentka pro helpdesk ve společnosti, která nabízí penzijní spoření."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()
    answer += "\nMohu Vám ještě s něčím pomoci?"
    if len(answer) > 300:
        answer = answer.rsplit('.', 1)[0] + '.'
    return answer

@app.route("/", methods=["GET"])
def home():
    return "Aplikace běží!"

@app.route("/api/chat", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "Dotaz nesmí být prázdný"}), 400
    
    if not collection.count():  # Oprava kontroly, zda je kolekce prázdná
        create_embeddings("./documents")
    
    context_documents = query_chromadb(query)
    answer = generate_answer_with_assistant(query, context_documents) if context_documents else "Bohužel, odpověď ve své databázi nemám."
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True, workers=2)
