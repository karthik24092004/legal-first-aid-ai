from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss
import pickle
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vectorstore/legal_index.faiss")

with open("vectorstore/chunks.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
sources = data["sources"]


# ---------------------------
# Intent Classifier
# ---------------------------
def classify_intent(query):
    query = query.lower()

    # ---------------------------
    # High-priority rules
    # ---------------------------

    # Online order / consumer dispute should win over "phone" or "disappeared"
    if (
        "ordered" in query
        or "online order" in query
        or "seller" in query
        or "not delivered" in query
        or "delivery" in query
        or "refund" in query
        or "marketplace" in query
    ):
        return "consumer"

    # Job scam rule
    if (
        "job" in query
        and (
            "fee" in query
            or "registration" in query
            or "training" in query
            or "payment" in query
            or "paid" in query
            or "blocked" in query
            or "recruiter" in query
        )
    ):
        return "fraud"

    # Blackmail / threat rule
    if (
        ("threat" in query or "threatening" in query or "blackmail" in query or "leak" in query)
        and ("photo" in query or "photos" in query or "video" in query or "money" in query or "upi" in query)
    ):
        return "harassment"

    identity_keywords = [
        "fake profile", "fake account", "impersonation",
        "using my name", "using my photos", "identity theft",
        "someone created a profile", "pretending to be me"
    ]

    account_keywords = [
        "hacked", "hack", "account hacked", "instagram account",
        "facebook account", "gmail account", "whatsapp account",
        "unauthorized access", "login", "logged in", "password reset",
        "posting messages from my account"
    ]

    document_keywords = [
    "aadhaar",
    "aadhar",
    "pan card",
    "passport",
    "driving licence",
    "license",
    "document",
    "documents",
    "lost my aadhaar",
    "lost my pan",

    # Additional document-related phrases
    "complaint file",
    "complaint copy",
    "lost complaint",
    "lost file",
    "lost documents",
    "case file"
    ]

    employment_keywords = [
        "salary", "unpaid salary", "salary not paid",
        "not paying", "not paid", "payment not received",
        "stipend", "internship", "intern", "employer",
        "startup", "experience letter", "final settlement",
        "termination", "terminated", "wrongful termination",
        "offer letter", "joining letter", "hr", "payroll",
        "resignation"
    ]

    harassment_keywords = [
        "stalking", "following", "follow", "harass", "unsafe",
        "threat", "threatening", "blackmail", "extortion",
        "leak", "private photos", "private video", "social media"
    ]

    fraud_keywords = [
        "upi", "fraud", "scam", "payment", "transaction",
        "bank", "qr", "otp", "phishing",
        "job offer", "job registration", "registration fee",
        "training fee", "recruiter", "whatsapp recruiter",
        "part time job", "work from home", "blocked me",
        "job scam"
    ]

    property_keywords = [
        "landlord", "tenant", "rent", "deposit", "security deposit",
        "eviction", "illegal eviction", "house owner", "flat owner",
        "property damage", "damaged my car", "damaged my property",
        "repair cost", "refuses to pay", "rental agreement",
        "neighbor", "neighbour", "tree", "branches",
        "boundary", "garden", "encroachment"
    ]

    theft_keywords = [
        "bike", "vehicle", "two-wheeler", "scooter", "cycle",
        "stolen", "theft", "missing", "disappeared",
        "phone stolen", "laptop stolen", "mobile stolen"
    ]

    if any(word in query for word in identity_keywords):
        return "identity"

    if any(word in query for word in account_keywords):
        return "account"

    if any(word in query for word in document_keywords):
        return "document"

    if any(word in query for word in employment_keywords):
        return "employment"

    if any(word in query for word in harassment_keywords):
        return "harassment"

    if any(word in query for word in fraud_keywords):
        return "fraud"

    if any(word in query for word in property_keywords):
        return "property"

    if any(word in query for word in theft_keywords):
        return "theft"

    return "general"


# ---------------------------
# Source Category Mapper
# ---------------------------
def get_source_category(source):
    source = source.lower()

    if "account_hacking" in source:
        return "account"

    if "upi" in source or "fraud" in source or "bank" in source or "job_scam" in source:
        return "fraud"
    
    if "document_loss" in source:
        return "document"
    
    if "employment_issues" in source:
        return "employment"
    
    if "consumer_disputes" in source:
        return "consumer"
    
    if "land_property" in source or "property" in source:
        return "property"
    
    if "identity_theft" in source:
        return "identity"

    if (
        "stalking" in source
        or "harassment" in source
        or "blackmail" in source
        or "cyberbullying" in source
        or "extortion" in source
    ):
        return "harassment"

    if (
        "bike" in source
        or "theft" in source
        or "vehicle" in source
        or "phone_theft" in source
        or "laptop_theft" in source
    ):
        return "theft"

    

    return "general"


# ---------------------------
# Filtered Retrieval Function
# ---------------------------
def retrieve_documents(query, k=8, forced_intent=None):
    intent = forced_intent if forced_intent else classify_intent(query)

    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k * 4)

    results = []

    for score, idx in zip(distances[0], indices[0]):
        source = sources[idx]
        source_category = get_source_category(source)

        if intent == "general" or source_category == intent:
            results.append({
                "chunk": chunks[idx],
                "source": source,
                "score": float(score)
            })

        if len(results) == k:
            break

    return results, intent




# ---------------------------
# Main RAG Function
# ---------------------------
def ask_groq(question, forced_intent=None):
    docs, intent = retrieve_documents(
        question,
        k=8,
        forced_intent=forced_intent
    )

    if not docs:
        return {
            "answer": "Sorry, I could not find relevant legal information in the documents.",
            "intent": intent,
            "sources": [],
            "confidence": 0
        }

    avg_score = sum([doc["score"] for doc in docs]) / len(docs)
    confidence = round(avg_score * 100, 2)

    # Low retrieval score guardrail
    if avg_score < 0.25:
        return {
            "answer": """
👨‍⚖️ Lawyer Uncle Says

Hey, thanks for reaching out.

This does not seem to match the legal first-aid topics currently available in my documents. I don't want to guess and give you unreliable guidance.

Incident Summary

This information is not available in the provided context.

Immediate Steps

This information is not available in the provided context.

Evidence To Preserve

This information is not available in the provided context.

Authorities To Contact

This information is not available in the provided context.

Important Notes

This information is not available in the provided context.
""",
            "intent": intent,
            "sources": [],
            "confidence": confidence
        }

    context = ""

    for i, item in enumerate(docs):
        context += f"[Chunk {i+1} | Source: {item['source']}]\n{item['chunk']}\n\n"

    prompt = f"""
Context:
{context}

User Question:
{question}

Instructions:
You are a legal first-aid assistant. Your job is to provide immediate, practical legal guidance based ONLY on the provided context.

Rules:
- Use ONLY the provided context as the primary and most reliable source of information.
- Do NOT invent facts, laws, procedures, authorities, or steps that are not present in the context.
- If the context does not contain enough information, clearly say:
  "This information is not available in the provided context."
- Do NOT mix information from unrelated incidents.
- Do NOT guess missing words or complete incomplete sentences from the context.
- Do NOT include authorities that are not relevant to the user's incident.
- Place each point only under the most relevant heading.
- Do not repeat the same idea in multiple sections.
- Evidence items must appear only under "Evidence To Preserve", not under "Immediate Steps".
- Consider the entire user query carefully.
- If the user indicates money was already paid, reflect that in the Incident Summary and Immediate Steps.

Output Rules:
- Keep the response structured and concise.
- Maintain a professional legal advisory tone.
- Do not add unnecessary explanations.

Lawyer Uncle Personality:
- Start every answer with "👨‍⚖️ Lawyer Uncle Says"
- Start with "Hey, thanks for reaching out."
- Speak like an experienced lawyer explaining things to a younger family member.
- Use natural, simple language.
- Give a short 3 - 4 sentence intro before the structured answer.
- Briefly identify the issue.
- Mention one practical observation about the situation.
- Focus the user on the most important immediate concern.
- Do not sound robotic, dramatic, or like customer support.
- Do not guarantee outcomes.

Strict Output Format:

👨‍⚖️ Lawyer Uncle Says

[Short natural introduction]

Incident Summary

Immediate Steps

Evidence To Preserve

Authorities To Contact

Important Notes
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    used_sources = list(set([doc["source"] for doc in docs]))

    return {
        "answer": answer,
        "intent": intent,
        "sources": used_sources,
        "confidence": confidence
    }