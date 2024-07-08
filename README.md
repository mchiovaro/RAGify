# RAGify

_A Gradio App for Retrieval-Augmented-Generation on PDFs_

**RAGify** is a tool for querying PDFs on the fly. PDFs are uploaded from the host computer, subject to OCR, vectorized using `langchain.text_splitter` functions, and stored in a temporary vector database, which is searched using `chroma`'s query function. Results are fed to the user's model of choice (currently either [tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) and [llama-2-7b-chat.Q5_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)).

## Instructions
1. Download and add desired LLMs to `models/`. \ _Note: Current version of the app accepts [tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) or [llama-2-7b-chat.Q5_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)._
2. Add the prompt templates in the `rag()` function and in the dropdown menu (via the `models` list).
3. Load dependencies
4. Run the app

## Contributions
Feel free to submit a pull request for any issues or improvements! \
Author: Megan Chiovaro (@mchiovaro)
