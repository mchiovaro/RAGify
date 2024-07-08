# %% imports
import gradio as gr
from pathlib import Path
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from uuid import uuid4
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
import mlflow
import mlflow.pyfunc

# %% initialize MLflow
#mlflow.set_tracking_uri(uri="http://0.0.0.0:8080")
mlflow.set_experiment("RAG on uploaded PDFs using local LLMs")

# %% set up the model
dir_ = Path(__file__).parent
def setup_model(model_name):
    
    # get file path
    model_path = f"{dir_}/models/{model_name}"

    # set model params
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.25,
        max_tokens=2000,
        n_ctx=2000, # max context size (i.e., prompt tokens plus generated tokens)
        top_p=1,
        callback_manager=callback_manager,
        verbose=True, # required to pass to CallbackManager
    )

    # log model parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("model_path", model_path)
    mlflow.log_param("temperature", 0.25)
    mlflow.log_param("max_tokens", 2000)
    mlflow.log_param("n_ctx", 2000)
    mlflow.log_param("top_p", 1)

    return llm

# %% tokenize
def tokenize(text_raw:str, model_max_chunk_length:int = 256) -> list:

    # chroma default model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    # max length: 256 char
    character_splitter = RecursiveCharacterTextSplitter(
        separators=['\n    \n', '\n\n', '\n', '. '],
        chunk_size=1000,
        chunk_overlap=0,
    )
    text_splitted = character_splitter.split_text('\n\n'.join(text_raw))

    # create tokens
    token_splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=model_max_chunk_length,
        model_name="all-MiniLM-L6-v2",
        chunk_overlap=0
    )
    text_tokens = []
    for text in text_splitted:
        text_tokens.extend(token_splitter.split_text(text))

    return text_tokens

# %% create vector database
def vdb(text_tokens):

    embedding_fn = SentenceTransformerEmbeddingFunction()
    chroma_db = chromadb.Client()
    # create unique collection identifier
    collection_name = str(uuid4())
    chroma_collection = chroma_db.create_collection(f"{collection_name}", embedding_function=embedding_fn)
    ids = [str(uuid4()) for _ in range(len(text_tokens))]
    chroma_collection.add(documents=text_tokens, ids=ids)

    return chroma_collection

# %% perform rag
def rag(file, question, model_name):

    # start tracking
    mlflow.start_run()

    # perform ocr
    reader = PdfReader(file)
    doc_texts = [page.extract_text().strip() for page in reader.pages]
    # Log the number of pages processed
    mlflow.log_metric("num_pages", len(reader.pages))
    mlflow.log_param("file_name", file.name)

    # tokenize the text
    text_tokens = tokenize(doc_texts)
    
    # create temp vdb
    chroma_collection = vdb(text_tokens)
    
    # query the vdb
    res = chroma_collection.query(query_texts=[question], n_results=2)
    docs = res["documents"][0]
    docs = ';'.join([f'{doc}' for doc in docs])

    # log the documents retrieved
    mlflow.log_text(docs, "retrieved_docs.txt")

    # set up the model
    llm = setup_model(model_name)

    # set system and user prompt
    system_prompt = f"You are answering this question: {question}. Use only the following information to answer: {docs}. Think step-by-step."
    user_prompt = f"{question}"

    # set up prompt correctly based on model being used
    if model_name == 'llama-2-7b-chat.Q5_K_M.gguf':
        prompt_template = f"<s>[INST]<<SYS>>\n{system_prompt}<</SYS>>\n{user_prompt}[/INST]"
        response = llm(prompt_template)
        # log the prompt
        mlflow.log_text(prompt_template, "prompt.txt")
    
    elif model_name == 'tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf':
        prompt_template = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
        response = llm(prompt_template)
        # log the prompt
        mlflow.log_text(prompt_template, "prompt.txt")

    else:
        response = "The selected model does not exist in the models folder. Please download the GGUF file and add it to this directory."
    
    # Log the response and end run
    mlflow.log_text(response, "response.txt")
    mlflow.end_run()

    return response

# %% create interface
model = ["llama-2-7b-chat.Q5_K_M.gguf", "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"]
interface = gr.Interface(
    fn=rag,
    inputs=[
        "file",
        gr.Textbox(label="Ask a Question"),
        gr.Dropdown(choices=model, label="Model", value='tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf')
    ],
    outputs=["text"],
    examples=[
        [str(dir_ / "example-pdfs/Python_source_stock_market_data.pdf"), 
         "How do I access stock market data using Python?", 
         "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"], 
        ],
    title="RAG on PDFs using a local LLM",
    description="Upload a file and ask a question."
)
interface.launch()

# %%