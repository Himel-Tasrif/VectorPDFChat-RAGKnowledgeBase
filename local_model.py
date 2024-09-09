import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_directory = "./mistral_model"  # Directory to save the model

# Hugging Face Access Token
hf_access_token = "your hugguing face token"  # Replace with your actual token

# Load the model and tokenizer using the access token
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    cache_dir=model_directory,
    token=hf_access_token  # Pass the access token for authentication
)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", 
    cache_dir=model_directory,
    token=hf_access_token  # Pass the access token for authentication
)

print(f"Model and tokenizer downloaded and saved in: {model_directory}")