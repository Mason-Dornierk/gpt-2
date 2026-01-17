from transformers import GPT2Tokenizer, GPT2Model
import torch #tensor math engine

# 1. Load GPT2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")#tokenizer converts text into token IDS
model = GPT2Model.from_pretrained("gpt2")#Model loads pretrained weights
model.eval()  # set model to evaluation mode

# 2. Your email text
email_text = "This is a test email to check AI detection."

# 3. Tokenize the email
inputs = tokenizer(email_text, return_tensors="pt")

# 4. Run through GPT2 to get hidden states
with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state  # shape: [1, seq_len, hidden_dim]

# 5. Pool over tokens to get a single vector
email_embedding = hidden_states.mean(dim=1)  # shape: [1, 768]

print("Email embedding shape:", email_embedding.shape)
print(email_embedding)  # optional: see the actual vector
