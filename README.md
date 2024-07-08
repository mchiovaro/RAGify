---
title: GradioRAG
emoji: 📊
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 4.28.2
app_file: app.py
pinned: false
---

A Gradio interface for performing RAG on uploaded PDFs.

Currently set up for models accessed with llama-cpp-python.

To use:
1. Create models/ subdirectory
2. Download desired model and add to models/. Note: Current version of the app accepts llama-2-7b-chat.Q5_K_M.gguf and tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf
3. Load dependencies using requirements.txt
4. Run app.py

Author: Megan Chiovaro (@mchiovaro)
