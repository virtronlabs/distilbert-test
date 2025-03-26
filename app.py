import torch
from transformers import DistilBertTokenizer, DistilBertModel
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    text = request.form['text']

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Get the model output (embedding)
    with torch.no_grad():
        outputs = model(**inputs)

    # The embeddings are in the 'last_hidden_state' (shape: [batch_size, sequence_length, hidden_size])
    embeddings = outputs.last_hidden_state

    # You can use the embedding of the [CLS] token as a representation of the entire sentence
    sentence_embedding = embeddings[:, 0, :].squeeze().tolist()  # Convert to list for JSON serialization

    return jsonify({'embedding': sentence_embedding})

if __name__ == '__main__':
    app.run(debug=True)
