import chromadb
import csv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

client = chromadb.PersistentClient(path="chroma.db")

# specify the embedding function to use
embedder = SentenceTransformerEmbeddingFunction(
    model_name="thenlper/gte-small",
    # specify the device to use, e.g. "cuda" or "cpu"
    device="cuda",
)

collection = client.get_or_create_collection(name="cars", embedding_function=embedder)

with open("data.csv") as f:
    reader = csv.DictReader(f)
    data = list(reader)

documents = [item["title"] for item in data]
ids = [str(i) for i in range(len(documents))]

if len(documents) != len(data) or len(documents) != len(ids):
    raise ValueError("Documents, data, and ids must have the same length")

collection.add(documents=documents, metadatas=data, ids=ids)
