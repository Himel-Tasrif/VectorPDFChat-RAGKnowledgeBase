import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load model and tokenizer from the local directory
model_directory = "./mistral_model"  # Directory where the model and tokenizer are saved

# Load the saved model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Use CPU instead of GPU
device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=torch.float16).to(device)  # Move model to CPU

# Function to generate responses based on user prompts
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to CPU
    outputs = model.generate(inputs.input_ids, max_length=200, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create Gradio chat interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Chat with the Mistral Model")
    msg = gr.Textbox(label="Your prompt")
    submit_btn = gr.Button("Send")

    def respond(message, chat_history):
        response = generate_response(message)
        chat_history.append((message, response))
        return chat_history, ""

    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])

# Run the Gradio interface
demo.launch()
