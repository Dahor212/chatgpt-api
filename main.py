from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import openai
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

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

# Nastavení OpenAI API klíče
openai.api_key = os.getenv("OPENAI_API_KEY")

# Kořenový endpoint (pro testování připojení)
@app.get("/")
def root():
    return {"message": "API je online! Použijte endpoint /ask pro odeslání dotazu."}

# Endpoint pro favicon.ico (pokud máte soubor v adresáři "static")
@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")

# Model pro požadavky na API
class QueryRequest(BaseModel):
    query: str

# Endpoint pro zpracování dotazů na /ask
@app.post("/ask")
async def ask(request: QueryRequest):
    query = request.query
    
    try:
        # Použití nové metody pro volání GPT-3.5 nebo GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Nebo použijte "gpt-4" pro nový model
            messages=[
                {"role": "user", "content": query}
            ]
        )
        answer = response['choices'][0]['message']['content'].strip()
        return {"answer": answer}
    except Exception as e:
        print(f"Chyba při zpracování dotazu: {str(e)}")  # Přidání logování pro chybové hlášky
        raise HTTPException(status_code=500, detail="Chyba při zpracování dotazu: " + str(e))

# Nastavení správného portu pro Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Použije port, který je přidělen Renderem
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
