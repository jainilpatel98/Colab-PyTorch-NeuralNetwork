import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


@st.cache_resource
def load_qa_model():
    # Load the official BERT model fine-tuned on SQuAD
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model


# Load model and tokenizer only once
tokenizer, model = load_qa_model()

st.title("BERT Question Answering System")
st.write("Enter a context passage and a question to get an answer using official BERT model")


def get_answer(question, context):
    # Tokenize inputs
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get start/end scores
    start_scores = torch.nn.functional.softmax(outputs.start_logits, dim=-1)
    end_scores = torch.nn.functional.softmax(outputs.end_logits, dim=-1)

    # Get the most likely positions
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1  # Add 1 to include end token

    # Calculate confidence score
    confidence = (start_scores[0, start_index] * end_scores[0, end_index - 1]).item()

    # Decode answer
    answer = tokenizer.decode(
        inputs.input_ids[0][start_index:end_index],
        skip_special_tokens=True
    )

    return answer, confidence


# Streamlit UI
context_input = st.text_area("Context:",
                             "Big data analytics involves examining large data sets to uncover hidden patterns, correlations, market trends, and customer preferences.")
question_input = st.text_input("Question:", "What does big data analytics involve?")

if st.button("Get Answer"):
    if context_input.strip() and question_input.strip():
        answer, confidence = get_answer(question_input, context_input)
        st.write(f"**Answer:** {answer}")
        st.write(f"**Confidence Score:** {round(confidence * 100, 2)}%")
    else:
        st.write("⚠️ Please provide both context and question!")