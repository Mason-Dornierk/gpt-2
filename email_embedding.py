from transformers import GPT2Tokenizer, GPT2Model
import torch

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")  # Note: GPT2Model, not GPT2LMHeadModel
model.eval()  # Set to evaluation mode

# Your email text
email_text = "This is a test email to check AI detection."

# Tokenize
inputs = tokenizer(email_text, return_tensors="pt")

# Get hidden states (semantic embeddings)
with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state  # shape: [1, seq_len, hidden_dim]

print("Hidden states shape:", hidden_states.shape)
