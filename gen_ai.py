import chromadb
from google import genai
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

# Gemini Client
client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("plant_knowledge")

history = []

# -------------------------
# Memory
# -------------------------
def update_memory(user, ai):
    history.append({
        "user": user,
        "ai": ai
    })

    if len(history) > 5:
        history.pop(0)


# -------------------------
# Intent Detection
# -------------------------
def detect_intent(text):
    text = text.lower().strip()

    question_words = ["what", "why", "how", "when", "where"]
    greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon"]

    is_greeting = any(re.search(rf"\b{word}\b", text) for word in greeting_words)

    is_question = (
        "?" in text or
        any(word in text for word in question_words)
    )

    # mixed intent → LLM
    if is_greeting and is_question:
        return detect_intent_llm(text)

    # greeting only
    if is_greeting:
        return "greeting"

    # question only
    if is_question:
        return "question"

    # fallback
    return detect_intent_llm(text)

def detect_intent_llm(text):
    prompt = f"""
Classify intent into ONLY one:
greeting, question, unknown

Return ONLY JSON:
{{"intent":"question"}}

Text:
{text}
"""

    try:
        response = client_ai.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )

        raw = response.text.strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start == -1 or end == 0:
            return "unknown"

        data = json.loads(raw[start:end])
        return data.get("intent", "unknown")

    except:
        return "unknown"

# -------------------------
# Router
# -------------------------
def route(intent):

    if intent == "greeting":
        return "direct"

    if intent == "question":
        return "rag"

    return "fallback"


# -------------------------
# Retrieve Knowledge
# -------------------------
def retrieve(prompt):
    results = collection.query(
        query_texts=[prompt],
        n_results=3,
        include=["documents", "distances"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    # 没有相关知识
    valid_docs = []

    for doc, dist in zip(documents, distances):
        if dist < 0.75:
            valid_docs.append(doc)

    if not valid_docs:
        return None

    return "\n\n".join(valid_docs)




# -------------------------
# Chat
# -------------------------
def chat(prompt):

    intent = detect_intent(prompt)
    mode = route(intent)

    if mode == "direct":
        return "Hello! I am your plant biology assistant."
    
    context = retrieve(prompt)

    if context is None:
        context = "No relevant knowledge found. Answer generally."

    memory = ""

    memory = "\n".join([
        f"User: {h['user']}\nAI: {h['ai']}"
        for h in history
    ])


    full_prompt = f"""
You are a strict plant biology assistant.

RULES:
- Use ONLY the KNOWLEDGE provided below.
- Do NOT use your own knowledge.
- If the answer is not in KNOWLEDGE, say:
  "I don't know based on the provided knowledge base."

KNOWLEDGE (use only this):
-------------------------
{context}
-------------------------

QUESTION:
{prompt}

IMPORTANT:
- Do not guess.
- Do not assume.
- Do not hallucinate.
"""
    print(context)
    response = client_ai.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=full_prompt
    )
    
    answer = response.text

    update_memory(prompt, answer)

    return answer


# -------------------------
# Main Loop
# -------------------------
while True:

    msg = input("You: ")

    if msg.lower() == "exit":
        break

    answer = chat(msg)

    print("AI:", answer)