import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Sample input text
text = "Hello, my dog is cute"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Get the model output (embedding)
with torch.no_grad():
    outputs = model(**inputs)

# The embeddings are in the 'last_hidden_state' (shape: [batch_size, sequence_length, hidden_size])
embeddings = outputs.last_hidden_state

# You can use the embedding of the [CLS] token as a representation of the entire sentence
sentence_embedding = embeddings[:, 0, :].squeeze()

# Print the sentence embedding (vector)
print(sentence_embedding)
