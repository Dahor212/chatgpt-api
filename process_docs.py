import os
import openai
import chromadb
from docx import Document

openai.api_key = os.getenv("OPENAI_API_KEY")



# Funkce na načtení textu z DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Vytvoření embeddingů a uložení do ChromaDB
def create_embeddings_from_docs(documents_path="documents"):
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="documents")

    for filename in os.listdir(documents_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(documents_path, filename)
            text = extract_text_from_docx(file_path)
            embedding = openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]

            collection.add(
                ids=[filename],
                embeddings=[embedding],
                metadatas=[{"source": filename}]
            )

    print("Embeddingy vytvořeny a uloženy do ChromaDB.")

if __name__ == "__main__":
    create_embeddings_from_docs()
