import faiss
import numpy as np
from sentence_transformers import SentenceTransformer 
import json

def load_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings, texts 


def build_and_save_index(embeddings: np.ndarray, index_path: str):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index


def save_id_mapping(chunks, mapping_path: str):
    id_to_text = {i: chunk for i, chunk in enumerate(chunks)}
    with open(mapping_path, 'w', encoding="utf-8") as f:
        json.dump(id_to_text, f)



chunks = load_chunks("data/processed_passages.jsonl")
embeddings, texts = embed_chunks(chunks)
build_and_save_index(embeddings, "data/knowledge_index.faiss")
save_id_mapping(chunks, "data/id_to_text.json")