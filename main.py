from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import openai
import requests
import os

# Nastavení API klíče OpenAI (nahraď vlastním klíčem)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializace FastAPI aplikace
app = FastAPI()

# Stáhne databázi z GitHubu a uloží ji lokálně
GITHUB_RAW_URL = "https://github.com/Dahor212/chatgpt-api/tree/751a9f7f74a7e510957b660dd70267fe0850fcda/chroma_db"
LOCAL_DB_PATH = "chromadb.db"

def download_chromadb():
    response = requests.get(GITHUB_RAW_URL)
    if response.status_code == 200:
        with open(LOCAL_DB_PATH, "wb") as f:
            f.write(response.content)
        print("ChromaDB stažena a uložena.")
    else:
        raise Exception("Nepodařilo se stáhnout ChromaDB z GitHubu.")

# Stáhnout databázi při spuštění serveru
download_chromadb()

# Připojení k lokální instanci ChromaDB
chroma_client = chromadb.PersistentClient(path=LOCAL_DB_PATH)
collection = chroma_client.get_collection(name="documents")

# Definice modelu pro dotazy
class Query(BaseModel):
    question: str

@app.post("/query/")
async def query_documents(query: Query):
    try:
        # Hledání nejbližšího embeddingu v ChromaDB
        results = collection.query(
            query_texts=[query.question],
            n_results=3  # Počet nejbližších výsledků
        )

        if not results or not results['documents'][0]:
            return {"response": "Údaje v databázi nejsou k dispozici."}

        # Spojení relevantních dokumentů do kontextu
        context = "\n".join(results['documents'][0])

        # Vytvoření promptu pro GPT
        prompt = f"Následující text obsahuje relevantní informace:\n\n{context}\n\nOdpověz na dotaz na základě těchto informací: {query.question}"

        # Zavolání OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Jsi užitečný asistent."},
                      {"role": "user", "content": prompt}]
        )

        return {"response": response["choices"][0]["message"]["content"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při zpracování dotazu: {str(e)}")

# Spuštění aplikace (pro lokální testování)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
