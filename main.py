import os
import shutil
import requests
import chromadb
from fastapi import FastAPI
from openai import OpenAI

# Nastavení cesty k databázi
DB_PATH = "./chroma.sqlite3"
GITHUB_DB_URL = "https://github.com/Dahor212/chatgpt-api/blob/main/chroma_db/chroma.sqlite3"

def download_db():
    """Stáhne databázi z GitHubu, pokud neexistuje."""
    if not os.path.exists(DB_PATH):
        print("Stahuji ChromaDB z GitHubu...")
        response = requests.get(GITHUB_DB_URL, stream=True)
        if response.status_code == 200:
            with open(DB_PATH, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print("ChromaDB stažena.")
        else:
            raise Exception("Nepodařilo se stáhnout databázi.")
    else:
        print("Databáze již existuje.")

# Stažení databáze při startu aplikace
download_db()

# Připojení ke stávající databázi
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection("documents")

# Inicializace FastAPI
app = FastAPI()

@app.get("/query/")
def query_chatbot(query: str):
    """Zpracuje dotaz a vyhledá odpověď v ChromaDB."""
    results = collection.query(query_texts=[query], n_results=3)
    if results and results['documents']:
        return {"response": results['documents'][0]}
    return {"response": "Omlouváme se, ale nemáme informace k tomuto dotazu."}
