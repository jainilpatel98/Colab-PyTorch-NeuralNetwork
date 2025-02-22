import transformers
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Load BERT model and tokenizer
attention_model_name = "bert-base-uncased"
attention_tokenizer = transformers.BertTokenizer.from_pretrained(attention_model_name)
attention_model = transformers.BertModel.from_pretrained(attention_model_name, output_attentions=True)


@app.route("/attention", methods=["POST"])
def attention_visualization():
    """Generate BERT attention visualization data layer-by-layer."""
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Text input is required!"}), 400

    inputs = attention_tokenizer(text, return_tensors="pt")
    tokens = attention_tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

    with torch.no_grad():
        outputs = attention_model(**inputs)

    attentions = torch.stack(outputs.attentions).squeeze().tolist()  # Shape: (num_layers, num_heads, seq_len, seq_len)

    return jsonify({
        "tokens": tokens,
        "num_layers": len(attentions),
        "attention_weights": attentions  # Contains all layers' attention weights
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)