import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

try:
    # Apply custom styling
    st.markdown(
        """
        <style>
        body {
            background-color: #e6f7ff;
        }
        .stApp {
            background-color: #e6f7ff;
        }
        .title {
            color: #4b0082;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load the DialoGPT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Initialize Streamlit app
    st.markdown('<div class="title">Enhanced Chatbot with DialoGPT</div>', unsafe_allow_html=True)
    st.write("Chat with the AI by typing your message below:")

    # Session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("You:", "", key="input")

    # Generate a response when the user enters a message
    if user_input:
        # Encode user input and append to chat history
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([torch.tensor([]).long()] + [new_input_ids], dim=-1)

        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
        )

        # Decode and store the bot's reply
        bot_reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        st.session_state.chat_history.append((user_input, bot_reply))

    # Display chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Bot🤖:** {bot_msg}")

except ModuleNotFoundError as e:
    print("Required module not found:", e)
    print("Please ensure you have installed all required packages: streamlit, transformers, and torch.")


