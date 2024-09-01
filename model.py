import gradio as gr
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint

from pathlib import Path
import chromadb
from unidecode import unidecode

from transformers import AutoTokenizer
import transformers
import torch
import tqdm
import accelerate
import re

# LLM model to use
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"


# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size, chunk_overlap):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits


# Create vector database
def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.PersistentClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
    )
    return vectordb


# Load vector database
def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding)
    return vectordb


# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    progress(0.5, desc="Initializing HF Hub...")
    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
        load_in_8bit=True,
    )

    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    progress(0.9, desc="Done!")
    return qa_chain


# Generate collection name for vector database
def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = collection_name.replace(" ", "-")
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    print('Filepath: ', filepath)
    print('Collection name: ', collection_name)
    return collection_name


# Initialize database
def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()):
    list_file_path = [x.name for x in list_file_obj if x is not None]
    progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(list_file_path[0])
    progress(0.25, desc="Loading document...")
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    progress(0.5, desc="Generating vector database...")
    vector_db = create_db(doc_splits, collection_name)
    progress(0.9, desc="Done!")
    return vector_db, collection_name, "Complete!"


def initialize_LLM(llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    print("LLM model: ", llm_model)
    qa_chain = initialize_llmchain(llm_model, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history


def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    new_history = history + [(message, response_answer)]
    return qa_chain, gr.update(
        value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page


def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    return list_file_path


def demo():
    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()

        gr.Markdown(
            """<center><h2>PDF-based chatbot</center></h2>
            <h3>Ask any questions about your PDF documents</h3>""")
        gr.Markdown(
            """<b>Note:</b> This AI assistant, using Langchain and open-source LLMs, performs retrieval-augmented generation (RAG) from your PDF documents. \
            The user interface explicitly shows multiple steps to help understand the RAG workflow. 
            This chatbot takes past questions into account when generating answers (via conversational memory), and includes document references for clarity purposes.<br>
            <br><b>Warning:</b> This space uses the free CPU Basic hardware from Hugging Face. Some steps and LLM models used below (free inference endpoints) can take some time to generate a reply.
            """)

        with gr.Tab("Step 1 - Upload PDF"):
            with gr.Row():
                document = gr.Files(height=100, file_count="multiple", file_types=["pdf"], interactive=True,
                                    label="Upload your PDF documents (single or multiple)")

        with gr.Tab("Step 2 - Process document"):
            with gr.Row():
                db_btn = gr.Radio(["ChromaDB"], label="Vector database type", value="ChromaDB", type="index",
                                  info="Choose your vector database")
            with gr.Accordion("Advanced options - Document text splitter", open=False):
                with gr.Row():
                    slider_chunk_size = gr.Slider(minimum=100, maximum=1000, value=600, step=20, label="Chunk size",
                                                  info="Chunk size", interactive=True)
                with gr.Row():
                    slider_chunk_overlap = gr.Slider(minimum=10, maximum=200, value=40, step=10, label="Chunk overlap",
                                                     info="Chunk overlap", interactive=True)
            with gr.Row():
                db_progress = gr.Textbox(label="Vector database initialization", value="None")
            with gr.Row():
                db_btn = gr.Button("Generate vector database")

        with gr.Tab("Step 3 - Initialize QA chain"):
            with gr.Row():
                slider_temperature = gr.Slider(minimum=0.01, maximum=1.0, value=0.7, step=0.1, label="Temperature",
                                               info="Model temperature", interactive=True)
            with gr.Row():
                slider_maxtokens = gr.Slider(minimum=224, maximum=4096, value=1024, step=32, label="Max Tokens",
                                             info="Model max tokens", interactive=True)
            with gr.Row():
                slider_topk = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="top-k samples",
                                        info="Model top-k samples", interactive=True)
            with gr.Row():
                llm_progress = gr.Textbox(value="None", label="QA chain initialization")
            with gr.Row():
                qachain_btn = gr.Button("Initialize Question Answering chain")

        with gr.Tab("Step 4 - Chatbot"):
            chatbot = gr.Chatbot(height=300)
            with gr.Accordion("Advanced - Document references", open=False):
                with gr.Row():
                    doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                    source1_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
                    source2_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
                    source3_page = gr.Number(label="Page", scale=1)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type message (e.g. 'What is this document about?')", container=True)
            with gr.Row():
                submit_btn = gr.Button("Submit message")
                clear_btn = gr.ClearButton([msg, chatbot], value="Clear conversation")

        db_btn.click(initialize_database, \
                     inputs=[document, slider_chunk_size, slider_chunk_overlap], \
                     outputs=[vector_db, collection_name, db_progress])
        qachain_btn.click(initialize_LLM, \
                          inputs=[slider_temperature, slider_maxtokens, slider_topk, vector_db], \
                          outputs=[qa_chain, llm_progress]).then(lambda: [None, "", 0, "", 0, "", 0], \
                                                                 inputs=None, \
                                                                 outputs=[chatbot, doc_source1, source1_page,
                                                                          doc_source2, source2_page, doc_source3,
                                                                          source3_page], \
                                                                 queue=False)

        msg.submit(conversation, \
                   inputs=[qa_chain, msg, chatbot], \
                   outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3,
                            source3_page], \
                   queue=False)
        submit_btn.click(conversation, \
                         inputs=[qa_chain, msg, chatbot], \
                         outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page,
                                  doc_source3, source3_page], \
                         queue=False)
        clear_btn.click(lambda: [None, "", 0, "", 0, "", 0], \
                        inputs=None, \
                        outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3,
                                 source3_page], \
                        queue=False)
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    demo()
