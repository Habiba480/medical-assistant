#  Medical Assistant Chatbot

** Medical Assistant** is an intelligent, interactive healthcare chatbot built with Streamlit and powered by advanced Retrieval-Augmented Generation (RAG) techniques. It helps users understand their symptoms, assess possible medical conditions, and receive reliable health guidance based on a structured and enriched knowledge base.

---

## Features

- **Semantic Symptom Matching**  
  Utilizes Sentence Transformers and FAISS to identify relevant conditions based on user input.

- **Advanced Medical Knowledge Base**  
  Covers multiple diseases with detailed attributes including symptoms, precautions, differential diagnoses, ICD-10 codes, risk factors, and complications.

- **Structured Response Generation**  
  Provides responses with clearly defined confidence levels, severity classifications, symptom match scores, and medical recommendations.

- **Context-Aware Chat**  
  Maintains multi-turn conversations with dynamic context integration for more accurate diagnosis.

- **Multi-Factor Ranking System**  
  Combines semantic similarity and symptom pattern recognition to improve relevance of medical suggestions.

---
##  Project Structure
```
Medical Assistant/
│
├── app.py                      # Main Streamlit app
├── requirements.txt            # Dependencies
├── README.md                   # Project description
│
└── src/
    ├── __init__.py             # (optional) make src importable
    ├── constants.py            # Prompts, config values
    ├── chat_state.py           # Chat session management
    ├── llm_client.py           # Handles requests to the LLM API
    ├── ui.py                   # Sidebar + chat rendering
                 # Sidebar + chat rendering
```
## Technologies Used

- Python 3.8+
- Streamlit – Frontend interface
- SentenceTransformers – Semantic embedding generation
- FAISS – Efficient vector similarity search
- Pandas & NumPy – Data processing
- Regex & Logging – NLP and debugging utilities

---


---

## How It Works

1. **Data Processing**  
   Loads a structured medical dataset and extracts key features such as symptoms, complications, duration, and risk factors.

2. **Indexing**  
   Embeds all condition profiles using `all-MiniLM-L6-v2` and creates a FAISS vector index for semantic search.

3. **Query Analysis**  
   Analyzes user queries to extract symptoms, intent, and severity indicators using rule-based NLP.

4. **Semantic + Symptom Matching**  
   Matches queries against the knowledge base using both semantic similarity and token overlap in symptoms.

5. **Structured Medical Response**  
   Generates a `MedicalResponse` object with confidence, severity, suggestions, and educational disclaimers.

---

## Example Use Cases

- "I have a sore throat and mild fever"  
  → Diagnoses likely common cold or influenza, offers precautions, and when to seek care.

- "Feeling tired, thirsty, and urinating a lot"  
  → Suggests diabetes with moderate to high confidence, explains the chronic nature and long-term risks.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Habiba480/medical-assistant.git
cd medical-assistant-rag
```
