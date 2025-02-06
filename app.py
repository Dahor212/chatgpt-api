import os
import requests
import chromadb
import psutil
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import json  # Přidání pro ladění JSON dat

# Nastavení OpenAI API klienta
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicializace ChromaDB
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection_name = "dokumenty_kolekce"
collection = client_chroma.get_or_create_collection(name=collection_name)

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

# Načtení embeddingů a obsahu dokumentů z GitHubu
def load_embeddings_from_github():
    url = "https://raw.githubusercontent.com/Dahor212/chatgpt-api/refs/heads/main/Embeddingy/embeddings.json"
    response = requests.get(url)
    if response.status_code == 200:
        embeddings_data = response.json()
        print(f"Načtené embeddingy (ukázka): {json.dumps(embeddings_data)[:500]}")  # Prvních 500 znaků pro kontrolu
        return embeddings_data
    else:
        raise Exception(f"Chyba při načítání embeddingů: {response.status_code}")

# Generování embeddingu pro dotaz pomocí OpenAI
def generate_query_embedding(query):
    response = client.embeddings.create(input=[query], model="text-embedding-ada-002")
    return response.data[0].embedding

# Výpočet kosinové podobnosti mezi dvěma vektory
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

# Vyhledání relevantních dokumentů v ChromaDB
def query_chromadb(query, n_results=5):
    embeddings_data = load_embeddings_from_github()
    query_embedding = generate_query_embedding(query)
    results = []
    
    for doc_name, doc_embeddings in embeddings_data.items():
        if isinstance(doc_embeddings, list):  # Ověření, že obsahuje seznam embeddingů
            for embedding in doc_embeddings:
                similarity = cosine_similarity(query_embedding, embedding)
                results.append({
                    "document": doc_name,
                    "similarity": similarity
                })
        else:
            print(f"Varování: Neočekávaný formát embeddingů pro {doc_name}")
    
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results[:n_results]

# Generování odpovědi s využitím GPT-3.5-turbo
def generate_answer_with_assistant(query, context_documents):
    if not context_documents:
        return "Bohužel, odpověď ve své databázi nemám."

    context = "\n\n".join([doc['document'] for doc in context_documents])
    print(f"Použitý kontext pro dotaz: {context}")

    messages = [
        {"role": "system", "content": "Jsi asistentka pro helpdesk ve společnosti, která nabízí penzijní spoření. Tvoje odpovědi musí být založeny pouze na následujících informacích z dokumentů. Pokud žádná odpověď není v těchto dokumentech, odpověz: 'Bohužel, odpověď ve své databázi nemám.'"},
        {"role": "user", "content": f"Kontext dokumentů:\n{context}\n\nOtázka: {query}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Chyba při generování odpovědi: {str(e)}"

@app.route("/", methods=["GET"])
def home():
    return "Aplikace běží!"

@app.route("/api/chat", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "Dotaz nesmí být prázdný"}), 400

    context_documents = query_chromadb(query)
    answer = generate_answer_with_assistant(query, context_documents)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True, workers=2)
