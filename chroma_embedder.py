import chromadb
import csv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import tqdm

print("Loading db...")
client = chromadb.PersistentClient(path="chroma.db")

# specify the embedding function to use
print("Loading embedder...")
embedder = SentenceTransformerEmbeddingFunction(
    model_name="thenlper/gte-small",
    # specify the device to use, e.g. "cuda" or "cpu"
    device="cuda",
)

print("Creating/getting collection...")
collection = client.get_or_create_collection(name="cars", embedding_function=embedder)

print("Loading data...")
with open("data.csv") as f:
    reader = csv.DictReader(f)
    data = list(reader)

documents = [item["excerpt"] for item in data]

if len(documents) != len(data):
    raise ValueError("Documents, data, and ids must have the same length")

print("Adding documents...")
for i, doc in tqdm.tqdm(enumerate(documents), total=len(documents)):
    collection.add(ids=str(i), documents=doc, metadatas=data[i])
