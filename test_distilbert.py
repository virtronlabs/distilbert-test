import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load tokenizer & model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Test sentence
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# Get model prediction
with torch.no_grad():
    logits = model(**inputs).logits

# Get label
predicted_class_id = logits.argmax().item()
print("Predicted sentiment:", model.config.id2label[predicted_class_id])
