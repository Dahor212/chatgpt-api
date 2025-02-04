import os
import openai
import chromadb
import psutil  # Import knihovny pro sledování systémových prostředků
from docx import Document
import tiktoken
from flask import Flask, request, jsonify
from flask_cors import CORS

# Použití proměnné prostředí pro OpenAI API klíč
openai.api_key = os.getenv("OPENAI_API_KEY")  # Načítáme API klíč z prostředí

client = chromadb.Client()
collection_name = "dokumenty_kolekce"
collection = client.get_or_create_collection(name=collection_name)

app = Flask(__name__)
CORS(app)  # Povolení CORS pro komunikaci s frontendem

# Funkce pro sledování využití paměti
@app.route("/api/memory", methods=["GET"])
def memory_usage():
    mem = psutil.virtual_memory()  # Získání informací o RAM
    return jsonify({
        "total": mem.total / 1024**2,      # Celková RAM v MB
        "used": mem.used / 1024**2,        # Použitá RAM v MB
        "available": mem.available / 1024**2,  # Dostupná RAM v MB
        "percent": mem.percent             # Procento využití
    })

# Funkce pro načítání dokumentů
def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".docx"):
            doc_path = os.path.join(directory_path, filename)
            document = Document(doc_path)
            content = "\n".join([para.text for para in document.paragraphs if para.text.strip() != ""])
            documents.append((filename, content))
    return documents

# Funkce pro dělení textu
def split_text(text, max_tokens=1000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]

# Funkce pro vytvoření embeddingů a jejich uložení do ChromaDB
def create_embeddings(documents):
    for doc_name, content in documents:
        text_chunks = split_text(content, max_tokens=1000)
        embeddings = []
        ids = []
        metadatas = []
        documents_list = []

        for i, chunk in enumerate(text_chunks):
            response = openai.embeddings.create(input=[chunk], model="text-embedding-ada-002")
            embedding = response.data[0].embedding  # Nový přístup k embeddingu
            ids.append(f"{doc_name}_{i}")  # Každý chunk dostane unikátní ID
            embeddings.append(embedding)  # Přidáváme embedding
            metadatas.append({"source": doc_name})  # Přidáváme metadata
            documents_list.append(chunk)  # Přidáváme dokumenty

        # Přidání do ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_list
        )

# Funkce pro dotazování do ChromaDB
def query_chromadb(query, n_results=5):
    response = openai.embeddings.create(input=[query], model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding  # Nový přístup k embeddingu
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=["documents"])
    if "documents" in results:
        return results["documents"]
    else:
        return []

# Funkce pro generování odpovědi
def generate_answer_with_assistant(query, context_documents):
    context = "\n\n".join(str(doc) for doc in context_documents)
    prompt = f"""
    Jsi asistentka pro helpdesk ve společnosti, která nabízí penzijní spoření. Tvoje odpovědi musí být založeny pouze na následujících informacích z dokumentů. Pokud žádná odpověď není v těchto dokumentech, odpověz: 'Bohužel, odpověď ve své databázi nemám.'

    Kontext dokumentů:
    {context}

    Otázka: {query}
    Odpověď:
    """
    response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": "Jsi asistentka pro helpdesk ve společnosti, která nabízí penzijní spoření."},
        {"role": "user", "content": prompt}
    ], max_tokens=500, temperature=0.7)

    answer = response.choices[0].message.content.strip()
    answer += "\nMohu Vám ještě s něčím pomoci?"
    if len(answer) > 300:
        answer = answer.rsplit('.', 1)[0] + '.'
    return answer

# API cesta pro kořenovou URL
@app.route("/", methods=["GET"])
def home():
    return "Aplikace běží!"

# API cesta pro chat
@app.route("/api/chat", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Dotaz nesmí být prázdný"}), 400

    documents = load_documents_from_directory("./documents")  # Uprav tuto cestu, pokud soubory nejsou v tomto adresáři
    if len(collection.get()["documents"]) == 0:
        create_embeddings(documents)
    context_documents = query_chromadb(query)

    loading_message = "Načítám odpověď, prosím čekejte..."

    if context_documents:
        answer = generate_answer_with_assistant(query, context_documents)
    else:
        answer = "Bohužel, odpověď ve své databázi nemám."

    return jsonify({"loading_message": loading_message, "answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True)
