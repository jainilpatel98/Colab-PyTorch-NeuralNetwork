import streamlit as st
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    AutoModelForCausalLM, ViTFeatureExtractor,
    ViTForImageClassification, pipeline
)
import torch
from PIL import Image

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

def football_chat():
    st.header("⚽ AI Scout Chat")
    st.write("Discuss football strategies, player stats, and game analysis!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="chat_input")

    if user_input:
        # Generate response using DialoGPT
        tokenizer, model = models["chat"]
        new_input = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        response = model.generate(new_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        bot_reply = tokenizer.decode(response[:, new_input.shape[-1]:][0], skip_special_tokens=True)

        st.session_state.chat_history.append((user_input, bot_reply))

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
    st.markdown("⚡ Powered by BERT, ViT, and DialoGPT | 🏈 AI Scout Pro v1.0")


if __name__ == "__main__":
    main()