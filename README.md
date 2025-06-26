# Haor PDF Chatbot

A custom knowledge-based PDF query chatbot using RAG, a VectorDB, and the Mistral-7B-Instruct-v0.2 LLM. Built with Gradio for an interactive web interface.  

**Live Demo:** https://huggingface.co/spaces/himel06/Haor_PDF_Chatbot

---

## Features

- Extracts knowledge from PDFs using Retrieval-Augmented Generation (RAG).
- Stores embeddings in a VectorDB (e.g.Chromadb).
- Interfaces with `mistralai/Mistral-7B-Instruct-v0.2` LLM.
- Multiple approaches: different `.py` modules (e.g., `app.py`, `brain.py`) offering alternative workflows.
- Gradio-based web UI for an easy interactive experience.

---

## Requirements

Make sure you have Python 3.10+ installed, then clone and install dependencies:

```bash
git clone https://huggingface.co/spaces/himel06/Haor_PDF_Chatbot.git
```

```bash
cd Haor_PDF_Chatbot
```
```bash
pip install -r requirements.txt
```


