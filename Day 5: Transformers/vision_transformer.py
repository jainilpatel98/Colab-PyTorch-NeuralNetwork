from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from PIL import Image
import streamlit as st


# Cache model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    model_name = 'google/vit-large-patch16-384'  # Using smaller model for faster inference
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    return feature_extractor, model


def main():
    st.title("Vision Transformer Image Classifier")

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    # File upload section
    uploaded_file = st.file_uploader("Choose an image...",
                                     type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add classify button
        if st.button("Classify Image"):
            # Show loading spinner
            with st.spinner('Analyzing image...'):
                # Load model (cached)
                feature_extractor, model = load_model()

                # Preprocess and predict
                inputs = feature_extractor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()

                # Get label
                labels = model.config.id2label
                predicted_label = labels[predicted_class_idx]

                # Show results
                st.success(f"Prediction: {predicted_label}")
                st.session_state.processed = True


if __name__ == "__main__":
    main()