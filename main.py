from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


app = FastAPI()

client = chromadb.PersistentClient(path="chroma.db")
embedder = SentenceTransformerEmbeddingFunction(model_name="thenlper/gte-small")
collection = client.get_or_create_collection(name="cars", embedding_function=embedder)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/chat/{car_str}")
def get_car(car_str: str, n_results: int = 5):
    if len(car_str) < 3:
        return {"error": "Query string must be at least 3 characters long"}

    res = collection.query(
        query_texts=[car_str],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    return res["metadatas"][0]
