# Propaganda Detection in Climate Change Articles

This project focuses on detecting misinformation and propaganda in climate-related news articles using both traditional and modern Natural Language Processing (NLP) techniques. The final model integrates external scientific knowledge with language modeling to provide more accurate and explainable results.

---

## Project Overview

Climate misinformation is a growing concern, particularly in news media. This project explores different approaches to classify climate news articles as either **factual** or **propaganda**.

We experimented with:
- Manual annotation and feature engineering
- Traditional ML models (Logistic Regression, Random Forest)
- A Retrieval-Augmented Generation (RAG) pipeline using GPT-2 and FAISS

---

## Methods

###  Stage 1: Manual Annotation
- Manually labeled 150 articles as factual or propaganda
- Identified linguistic and emotional markers (e.g., readability, pronouns, superlatives, emotional tone)

###  Stage 2: Traditional NLP
- Preprocessing (lowercasing, tokenizing, stopword & punctuation removal)
- Feature extraction:
  - POS tags and ratio features
  - Emotion scores via `NRCLex`
  - Readability metrics
  - Spelling error rate
- Models used:
  - Logistic Regression
  - Random Forest (with and without simple knowledge base)

### Stage 3: Retrieval-Augmented Generation (RAG)
This section describes the most advanced method used in this project: a **Retrieval-Augmented Classification** pipeline that combines:

- **Fine-tuned GPT-2** for text classification  
- **SentenceTransformer embeddings** for semantic understanding  
- **FAISS vector search** for fast retrieval of scientific knowledge  
- A custom **RAG-style prompt engineering** workflow  

This hybrid method allows the model to evaluate climate news in the context of **verified scientific evidence**, leading to more accurate and robust classification.

<img width="975" height="511" alt="image" src="https://github.com/user-attachments/assets/482e6976-188d-4719-8804-b2ce7604c978" />

