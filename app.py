import os
import requests
import chromadb
import psutil
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

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

# Načtení embeddingů z GitHubu
def load_embeddings_from_github():
    url = "https://raw.githubusercontent.com/Dahor212/chatgpt-api/refs/heads/main/Embeddingy/embeddings.json"
    response = requests.get(url)
    if response.status_code == 200:
        embeddings_data = response.json()
        return embeddings_data
    else:
        raise Exception(f"Chyba při načítání embeddingů: {response.status_code}")

# Vyhledání relevantních dokumentů v ChromaDB na základě embeddingů z GitHubu
def query_chromadb(query, n_results=5):
    # Načteme embeddingy z GitHubu
    embeddings_data = load_embeddings_from_github()

    # Představme si, že query_embedding je již získaný embedding pro dotaz
    query_embedding = [0.1, 0.2, 0.3]  # Příklad embeddingu pro dotaz

    # Prohledáme embeddingy z GitHubu a vrátíme dokumenty s nejbližšími vektory
    results = []
    for doc_name, doc_embeddings in embeddings_data.items():
        for embedding in doc_embeddings:
            similarity = cosine_similarity(query_embedding, embedding)
            results.append({
                "document": doc_name,
                "similarity": similarity,
                "embedding": embedding
            })
    
    # Seřadíme výsledky podle podobnosti
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results[:n_results]

# Funkce pro výpočet kosinové podobnosti mezi dvěma vektory
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

# Generování odpovědi s využitím GPT modelu
def generate_answer_with_assistant(query, context_documents):
    if not context_documents:
        return "Bohužel, odpověď ve své databázi nemám."
    
    context = "\n\n".join([doc['document'] for doc in context_documents])
    prompt = f"""
    Jsi asistentka pro helpdesk ve společnosti, která nabízí penzijní spoření. Tvoje odpovědi musí být založeny pouze na následujících informacích z dokumentů. Pokud žádná odpověď není v těchto dokumentech, odpověz: 'Bohužel, odpověď ve své databázi nemám.'
    
    Kontext dokumentů:
    {context}
    
    Otázka: {query}
    Odpověď:
    """

    # Zavoláme OpenAI model pro generování odpovědi
    response = openai.Completion.create(
        model="text-davinci-003",  # nebo použijte jiný model dle potřeby
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )
    answer = response.choices[0].text.strip()
    
    # Pokud odpověď je příliš dlouhá, omezíme ji
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

    # Vyhledáme relevantní dokumenty z GitHubu
    context_documents = query_chromadb(query)
    
    # Generujeme odpověď na základě kontextu dokumentů
    answer = generate_answer_with_assistant(query, context_documents)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True, workers=2)
