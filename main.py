from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import openai
import chromadb
import requests
import json
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil

# Inicializace FastAPI aplikace
app = FastAPI()

# CORS konfigurace
origins = [
    "http://dotazy.wz.cz",  # Povolte pouze konkrétní doménu, ne "*" pro bezpečnost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Povolené domény
    allow_credentials=True,
    allow_methods=["*"],  # Povolit všechny HTTP metody (POST, GET, OPTIONS)
    allow_headers=["*"],  # Povolit všechny hlavičky
)

# Načtení OpenAI API klíče z prostředí
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Chybí API klíč OpenAI. Nastavte proměnnou prostředí OPENAI_API_KEY.")

# URL k GitHubu s ChromaDB
GITHUB_CHROMADB_URL = "https://github.com/Dahor212/chatgpt-api/blob/main/chroma_db/chroma.sqlite3"

# Stažení a rozbalení ChromaDB databáze z GitHubu
def download_chromadb_from_github(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
            # Předpokládáme, že je archiv ZIP
            shutil.unpack_archive(tmp_file_path, "chroma_db")
            os.remove(tmp_file_path)  # Odstraní dočasný soubor
            return "chroma_db"
    else:
        raise RuntimeError("Nepodařilo se stáhnout databázi ChromaDB z GitHubu.")

# Cesta k ChromaDB databázi (předpokládáme, že je po rozbalení ve složce 'chroma_db')
DB_PATH = "chroma_db"

# Pokud databáze ještě neexistuje, stáhneme ji z GitHubu
if not os.path.exists(DB_PATH):
    print("Stahuji ChromaDB z GitHubu...")
    DB_PATH = download_chromadb_from_github(GITHUB_CHROMADB_URL)
    print("ChromaDB byla úspěšně stažena a rozbalena.")

# Inicializace ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="docs")

# Kořenový endpoint (pro testování připojení)
@app.get("/")
def root():
    return {"message": "API je online! Použijte endpoint /ask pro odeslání dotazu."}

# Endpoint pro favicon.ico
@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")

# Model pro požadavky na API
class QueryRequest(BaseModel):
    query: str

# Funkce pro generování embeddingu dotazu pomocí OpenAI API
def generate_query_embedding(query: str):
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-ada-002"  # Používáme model pro embeddingy textu
    )
    # Opravený přístup k embeddingu
    return response['data'][0]['embedding']

# Endpoint pro zpracování dotazů na /ask
@app.post("/ask")
async def ask(request: QueryRequest):
    query = request.query

    try:
        # Generování embeddingu dotazu
        query_embedding = generate_query_embedding(query)

        # Hledání nejpodobnějších dokumentů v ChromaDB
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        
        if results["documents"]:
            relevant_docs = "\n".join(results["documents"][0])
        else:
            return {"answer": "Odpověď nelze najít v dostupných dokumentech."}
        
        # Použití OpenAI API s omezením na nalezené dokumenty
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Používáme nový způsob pro volání Completion modelu
            prompt=f"Na základě těchto dokumentů odpověz na dotaz:\n{relevant_docs}\n\nDotaz: {query}",
            max_tokens=1000
        )
        answer = response['choices'][0]['text'].strip()
        return {"answer": answer}
    
    except openai.OpenAIError as e:
        print(f"Chyba při volání OpenAI API: {str(e)}")
        raise HTTPException(status_code=500, detail="Chyba při komunikaci s OpenAI API.")
    
    except Exception as e:
        print(f"Neočekávaná chyba: {str(e)}")
        raise HTTPException(status_code=500, detail="Interní chyba serveru.")

# Spuštění aplikace
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Použije port, který je přidělen Renderem
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
