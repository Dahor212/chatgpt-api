import os
import openai
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Připojení k ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="documents")


class QueryRequest(BaseModel):
    query: str


@app.post("/ask")
def ask_question(request: QueryRequest):
    query_embedding = openai.Embedding.create(input=request.query, model="text-embedding-ada-002")["data"][0][
        "embedding"]

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    if results['documents']:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Odpovídej pouze na základě poskytnutých dokumentů."},
                {"role": "user", "content": results['documents'][0]}
            ]
        )
        return {"answer": response["choices"][0]["message"]["content"]}
    else:
        return {"answer": "Na tento dotaz nemám odpověď ve své databázi."}
