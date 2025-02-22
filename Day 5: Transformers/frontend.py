import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Backend API URL
BACKEND_URL = "http://127.0.0.1:5001"

st.header("BERT Attention Matrix Box")

# Initialize session state variables to store attention data
if "attention_data" not in st.session_state:
    st.session_state.attention_data = None
    st.session_state.tokens = None
    st.session_state.num_layers = None

# Text input
sample_text = st.text_area("Enter text for attention visualization:", "Machine learning is powerful.")

# Fetch attention data only when clicking the button
if st.button("Visualize Attention"):
    if sample_text:
        response = requests.post(f"{BACKEND_URL}/attention", json={"text": sample_text})

        if response.status_code == 200:
            data = response.json()
            st.session_state.attention_data = np.array(data["attention_weights"])  # Shape: (num_layers, num_heads, seq_len, seq_len)
            st.session_state.tokens = data["tokens"]
            st.session_state.num_layers = data["num_layers"]

            st.success("Attention Matrix Ready!")
        else:
            st.error("Error processing request.")
    else:
        st.warning("Please enter text to visualize.")

# Check if attention data is available
if st.session_state.attention_data is not None:
    tokens = st.session_state.tokens
    num_layers = st.session_state.num_layers
    attention_weights = st.session_state.attention_data

    # Select Layer and Head without losing previous data
    selected_layer = st.slider("Select Layer", 0, num_layers - 1, 0, key="layer_slider")
    selected_head = st.slider("Select Attention Head", 0, attention_weights.shape[1] - 1, 0, key="head_slider")

    # Extract attention matrix
    attention_matrix = attention_weights[selected_layer, selected_head]  # (seq_len, seq_len)

    # Plot attention matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title(f"Attention Matrix - Layer {selected_layer}, Head {selected_head}")
    st.pyplot(fig)