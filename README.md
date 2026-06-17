# 🚨 Legal First Aid AI

**Meet 👨‍⚖️ Lawyer Uncle — your AI-powered legal first-aid assistant.**

Before you call a lawyer, talk to Lawyer Uncle.

Legal First Aid AI helps users understand what to do immediately after common legal, cybercrime, employment, consumer, property, and fraud-related incidents. Users describe their situation in plain language, and the assistant provides structured first-aid guidance based on a curated legal knowledge base.

---

## 🌐 Live Demo

**Streamlit App:** https://legal-first-aid-ai-d6opedus7bp46t8g5fn9fm.streamlit.app/

---

## 📌 Problem Statement

Many people face situations such as:

* UPI fraud
* Account hacking
* Blackmail
* Job scams
* Employment disputes
* Bike or phone theft
* Consumer complaints
* Property disputes
* Document loss

Most users do not know:

* What steps to take immediately
* What evidence to preserve
* Which authorities to contact
* What mistakes to avoid

Legal First Aid AI provides quick and structured legal first-aid guidance during these critical moments.

---

## ✨ Features

### 👨‍⚖️ Lawyer Uncle Persona

Provides responses in a calm, practical, and easy-to-understand style.

### 🧠 Intent Classification

Automatically detects the category of the incident:

* Fraud
* Harassment
* Employment
* Consumer
* Property
* Theft
* Identity Theft
* Account Hacking
* Document Loss

### 🔍 Semantic Search

Uses sentence embeddings and FAISS vector search to retrieve the most relevant legal information.

### 📚 Retrieval-Augmented Generation (RAG)

Combines retrieved legal knowledge with an LLM to generate accurate and structured responses.

### 💬 Follow-Up Question Handling

Supports contextual follow-up questions such as:

User:

> My bike was stolen outside my college.

User:

> What evidence should I collect?

The assistant remembers the previous incident and answers appropriately.

### 📋 Structured Guidance

Responses are organized into:

* Incident Summary
* Immediate Steps
* Evidence To Preserve
* Authorities To Contact
* Important Notes

---

## 🏗️ Architecture

```text
User Query
     ↓
Intent Classification
     ↓
Semantic Embedding
     ↓
FAISS Vector Search
     ↓
Relevant Legal Documents
     ↓
Context Construction
     ↓
Groq LLM
     ↓
Lawyer Uncle Response
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### AI / NLP

* Groq LLM (Llama 3.3 70B Versatile)
* Sentence Transformers
* all-MiniLM-L6-v2

### Vector Database

* FAISS

### Backend

* Python

### Deployment

* Streamlit Community Cloud

---

## 📂 Project Structure

```text
Legal_first_aid/
│
├── app.py
├── rag_pipeline.py
├── ingest.py
├── requirements.txt
│
├── documents/
│   ├── account_hacking.md
│   ├── bank_fraud.md
│   ├── bike_theft.md
│   ├── blackmail.md
│   ├── consumer_disputes.md
│   ├── employment_issues.md
│   ├── identity_theft.md
│   ├── job_scam.md
│   ├── land_property_issues.md
│   └── ...
│
└── vectorstore/
    ├── legal_index.faiss
    ├── chunks.pkl
    └── document_mapping.pkl
```

---

## 🚀 How It Works

1. User describes an incident.
2. The system detects the most likely legal category.
3. Relevant legal documents are retrieved using semantic similarity search.
4. Retrieved context is passed to the Groq LLM.
5. Lawyer Uncle generates structured legal first-aid guidance.
6. Follow-up questions maintain context when possible.

---

## ⚠️ Disclaimer

Legal First Aid AI provides educational legal first-aid guidance only.

It is **not** a substitute for professional legal advice, legal representation, or consultation with a qualified lawyer.

Users should consult appropriate authorities or legal professionals for case-specific advice.

---

## 👨‍💻 Author

**Gurram Karthik Reddy**

B.Tech (Computer Science – Cyber Security)

Passionate about AI, GenAI applications, and problem-solving through technology.

---

## ⭐ Future Improvements

* Multi-intent incident detection
* Larger legal knowledge base
* Regional legal information
* Multilingual support
* Citation-based responses
* Conversation history and case tracking
* FastAPI backend API version
