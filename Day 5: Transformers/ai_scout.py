import streamlit as st
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    AutoModelForCausalLM, ViTFeatureExtractor,
    ViTForImageClassification, pipeline
)
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import streamlit as st
import os
# import google.generativeai as genai
# from google.generativeai import types  # Only if you need specific types
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --------------------------
# Custom Styling & Configuration
# --------------------------

st.set_page_config(page_title="AI Football Scout", page_icon="⚽", layout="wide")

# Apply football-themed styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #f0f2f6;
        border: 2px solid #2a5298;
    }
    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF5252;
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background-color: #1e3c72;
    }
    </style>
    """, unsafe_allow_html=True)


# --------------------------
# Model Loading (Cached)
# --------------------------

@st.cache_resource
def load_models():
    # QA Model
    qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    # Chat Model
    chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Vision Model
    vit_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-384')
    vit_model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-384')

    return {
        "qa": (qa_tokenizer, qa_model),
        "chat": (chat_tokenizer, chat_model),
        "vision": (vit_extractor, vit_model)
    }


models = load_models()


# --------------------------
# Core Functionality
# --------------------------

# def football_chat():
#     st.header("⚽ AI Scout Chat")
#     st.write("Discuss football strategies, player stats, and game analysis!")
#
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#
#     user_input = st.text_input("You:", key="chat_input")
#
#     if user_input:
#         # Generate response using DialoGPT
#         tokenizer, model = models["chat"]
#         new_input = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
#         response = model.generate(new_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
#         bot_reply = tokenizer.decode(response[:, new_input.shape[-1]:][0], skip_special_tokens=True)
#
#         st.session_state.chat_history.append((user_input, bot_reply))
#
#     for user_msg, bot_msg in st.session_state.chat_history[-5:]:
#         st.markdown(f"**You:** {user_msg}")
#         st.markdown(f"**Scout:** {bot_msg}")
#         st.markdown("---")

import streamlit as st
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configuration
GENAI_API_KEY = "AIzaSyBA6fPraMaU7PEiAiBbTL071gNAe95xsZs"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzUyMDA5MTQwfQ.BpKQ-ZrWfPPca_GB5v1enA8nZKI_bez2ahmD73sAcu4"
QDRANT_URL = "https://3d184921-7bd2-4bff-9699-8bbc5c2999f2.us-east4-0.gcp.cloud.qdrant.io"
COLLECTIONS = [
    "nfl_analyses",
    "reddit_comment_texts",
    "reddit_post_texts",
    "reddit_post_titles",
    "scouting_reports"
]


# Initialize clients with caching
@st.cache_resource
def load_clients():
    # Initialize Gemini client (keeping your original structure)
    gemini_client = genai.Client(api_key=GENAI_API_KEY)

    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Initialize encoder
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    return gemini_client, qdrant_client, encoder


def get_rag_context(query, qdrant_client, encoder):
    """Retrieve context from all collections without changing your flow"""
    try:
        query_embedding = encoder.encode(
            query,
            convert_to_tensor=False,
            normalize_embeddings=True
        ).tolist()

        context_parts = []

        for collection in COLLECTIONS:
            hits = qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=2,
                with_payload=True,
                timeout=15
            )

            coll_ctx = ""

            if hits:
                for i, hit in enumerate(hits, 1):
                    if hit.payload:
                        # Dynamic payload handling
                        for key, value in hit.payload.items():
                            # Handle lists and long text
                            if isinstance(value, list):
                                val_str = ", ".join(map(str, value))
                            else:
                                val_str = str(value)

                            # Truncate long values
                            coll_ctx += f"- **{key}**: {str(val_str)}\n"
                    else:
                        coll_ctx += "\n"

                    coll_ctx += "\n"
            else:
                coll_ctx += ""

            context_parts.append(coll_ctx)

        return "\n".join(context_parts)

    except Exception as e:
        st.error(f"Context retrieval error: {str(e)}")
        return ""


def generate_response(query, gemini_client, qdrant_client, encoder):
    """Generate response using your original client structure with RAG"""
    try:
        # Get context first
        context = get_rag_context(query, qdrant_client, encoder)

        # Build the prompt your way
        enhanced_query = f"""Analyze this football context and answer the question:

        {context}

        Question: {query}

        Provide a detailed professional analysis:"""

        # Use your original request structure
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=enhanced_query)]
            )  # Added missing closing parenthesis
        ]

        response = gemini_client.models.generate_content_stream(
            model="gemini-2.5-pro-exp-03-25",
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="text/plain")
        )

        full_text = ""
        for chunk in response:
            if chunk.candidates:  # Fixed indentation
                full_text += chunk.candidates[0].content.parts[0].text

        return full_text

    except Exception as e:
        return f"Error: {str(e)}"


def football_chat():
    st.header("⚽ AI Scout Chat (Enhanced with RAG)")
    st.write("Discuss football strategies, player stats, and game analysis!")

    # Initialize clients
    gemini_client, qdrant_client, encoder = load_clients()

    # Chat history - keeping your original structure
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="chat_input")

    if user_input:
        with st.spinner("Analyzing with RAG..."):
            bot_reply = generate_response(user_input, gemini_client, qdrant_client, encoder)

        # Keep your original chat history format
        st.session_state.chat_history.append((user_input, bot_reply))

    # Display history your way
    for user_msg, bot_msg in st.session_state.chat_history[-5:]:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Scout:** {bot_msg}")
        st.markdown("---")


def vision_analysis():
    st.header("📸 Play Analysis")
    st.write("Upload images of formations, player movements, or equipment")

    uploaded_file = st.file_uploader("Choose football image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        if st.button("Analyze Play"):
            extractor, model = models["vision"]
            inputs = extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            prediction = model.config.id2label[outputs.logits.argmax(-1).item()]

            st.success(f"**Analysis:** {football_interpretation(prediction)}")


def football_interpretation(prediction):
    football_keywords = {
        "helmet": "Player safety equipment detected",
        "field": "Football field layout analysis",
        "ball": "Ball position and trajectory estimation",
        "player": "Player positioning and movement analysis"
    }
    return next((v for k, v in football_keywords.items() if k in prediction.lower()), "Game situation analysis")


def document_qa():
    st.header("📄 Scouting Report Analysis")
    st.write("Analyze player reports, match summaries, and strategy documents")

    context = st.text_area("Paste document text:", height=150,
                           value="The quarterback demonstrated exceptional arm strength during Sunday's game, completing 78% of passes over 20 yards...")
    question = st.text_input("Ask about the document:")

    if st.button("Find Answer"):
        tokenizer, model = models["qa"]
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])

        st.markdown(f"**Key Insight:** {answer}")
        st.markdown(f"**Confidence:** {torch.max(outputs.start_logits).item() * 100:.1f}%")


# --------------------------
# App Layout
# --------------------------

def main():
    st.title("🤖 AI Football Scout Pro")
    st.markdown("## Your All-in-One Football Analysis Assistant")

    # Sidebar Navigation
    app_mode = st.sidebar.selectbox("Choose Mode", ["Chat Analysis", "Visual Play Breakdown", "Document Insights"])

    # Main Content
    if app_mode == "Chat Analysis":
        football_chat()
    elif app_mode == "Visual Play Breakdown":
        vision_analysis()
    elif app_mode == "Document Insights":
        document_qa()

    # Footer
    st.markdown("---")
    st.markdown("⚡ Powered by BERT, ViT, and DialoGPT, Gemini | 🏈 AI Scout Pro v1.0")


if __name__ == "__main__":
    main()