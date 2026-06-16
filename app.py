import streamlit as st
from rag_pipeline import ask_groq

st.set_page_config(
    page_title="Legal First Aid AI",
    page_icon="🚨",
    layout="centered"
)

# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🚨 Legal First Aid")

st.caption(
    "Meet 👨‍⚖️ Lawyer Uncle — your AI-powered legal assistant for everyday problems."
)

st.markdown(
    """
    > **Before you call a lawyer,  
    > talk to Lawyer Uncle.**
    """
)

# Show Clear Chat only after conversation starts
if st.session_state.messages:
    col1, col2 = st.columns([5, 1])

    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.messages = []
            st.rerun()

# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------
# Display Chat History
# ---------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):

        st.markdown(message["content"].replace("\n", "  \n"))

        if message["role"] == "assistant":
            intent = message.get("intent", "N/A")
            confidence = message.get("confidence", "N/A")
            sources = message.get("sources", [])

            st.caption(
                f"Category: {intent.title()} | Retrieval Similarity Score: {confidence}%"
            )

            with st.expander("Sources Used"):
                if sources:
                    for source in sources:
                        st.write(f"- {source}")
                else:
                    st.write("No source documents found.")


# ---------------------------
# Chat Input
# ---------------------------
user_input = st.chat_input("Describe your incident...")

if user_input:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Find previous assistant intent
    previous_intent = None

    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and "intent" in msg:
            previous_intent = msg["intent"]
            break

    # Detect short follow-up questions
    follow_up_phrases = [
        "what evidence",
        "what should i do",
        "what next",
        "who should i contact",
        "what authorities",
        "what documents",
        "how do i proceed",
        "what should i collect",
        "what proof",
        "what to preserve"
    ]

    is_follow_up = any(
        phrase in user_input.lower()
        for phrase in follow_up_phrases
    )

    # Prevent long detailed incidents from being treated as follow-ups
    is_short_follow_up = is_follow_up and len(user_input.split()) <= 8

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Retrieving legal guidance..."):

            previous_user_incident = ""

            for msg in reversed(st.session_state.messages):
                if msg["role"] == "user" and msg["content"] != user_input:
                    previous_user_incident = msg["content"]
                    break

            if is_short_follow_up and previous_intent and previous_user_incident:
                memory_query = f"""
Previous Incident:
{previous_user_incident}

Follow-up Question:
{user_input}
"""
                result = ask_groq(
                    memory_query,
                    forced_intent=previous_intent
                )
            else:
                result = ask_groq(user_input)

            answer = result["answer"]
            intent = result["intent"]
            sources = result["sources"]
            confidence = result["confidence"]

        st.info(f"Detected Category: {intent.title()}")
        st.success(f"Retrieval Similarity Score: {confidence}%")

        with st.expander("Sources Used"):
            if sources:
                for source in sources:
                    st.write(f"- {source}")
            else:
                st.write("No source documents found.")

        st.markdown(answer.replace("\n", "  \n"))

    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "intent": intent,
            "sources": sources,
            "confidence": confidence
        }
    )


# ---------------------------
# Clear Chat
# ---------------------------
# ---------------------------

st.caption(
    "⚠️ Legal first-aid guidance only. Not a substitute for professional legal advice."
)