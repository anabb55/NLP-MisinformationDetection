import fitz
from pathlib import Path
import re
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import json
import os

def load_pdf_file(pdf_path, skip_pages: int = 2) -> str:
    text = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(skip_pages, len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            if page_text.strip():
                text.append(page_text)

      
        return "\n".join(text)
    
    except Exception as e:
        print(f"Faied to load PDF {pdf_path.name}: {e}")
        return ""
    

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w.,!? %$€-]", "", text)
    text = re.sub(r'\n\d+\n', '\n', text) ##remove page numbers
    text = re.sub(r"(\d+(\.\d+)*)(,\s*\d+(\.\d+)*)+", "", text)
    text = re.sub(r"\b[A-Z]\.\d+(\.\d+)*", "", text)
    return text.strip()


def chunk_text(text, max_sentences=3) ->List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        chunks.append(chunk)
    return chunks

def create_chunks_with_metadata(text: str, source_id: str, max_sentence: int = 3) -> List[Dict]:
    sentence_chunks = chunk_text(text, max_sentences=max_sentence)
    return[
        {
            "id": f"{source_id}_chunk_{i}",
            "text": chunk,
            "source": source_id
        }
        for i, chunk in enumerate(sentence_chunks)
    ]


def save_chunks_to_jsonl(chunks: List[Dict], output_path) -> None:
    output_path.parent.mkdir(parents = True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f)
            f.write("\n")


file_path = "data/SRCCL_SPM.pdf"
source_id = os.path.splitext(os.path.basename(file_path))[0]
output_path = Path("data/processed_passages.jsonl")

raw_data = load_pdf_file(file_path)
cleaned_text = clean_text(raw_data)
chunks = create_chunks_with_metadata(cleaned_text, source_id, max_sentence=4)
save_chunks_to_jsonl(chunks,output_path)