# German RAG
This repo contains a RAG-based LLM which can answer German user queries by calling the Hggingface API services for inference.
There are two scripts in the repo:
1. **rag_agent.py**: Script for running the model locally in the terminal
2. **rag_agent.py**: Script to run the model with *Streamlit* and *localhost* ([Streamlit](https://docs.streamlit.io/))

## Prerequisites 
For the model run to run sucessfully, three requirements must be fulfilled:
1. **LangChain**: The scripts use *LangChain v1* for the model, embeddings and RAG setup ([Docs](https://docs.langchain.com/oss/python/langchain/overview))
2. **Huggingface** To use the scripts, the user needs a Huggingface API token in order to call the HF API for model inference
3. **requirements.txt**: the requirements.txt file must be installed to have all dependencies

## Model & Embeddings
The model used for the project is **SmolLM3-3B**, a 3B parameter LLM supporting reasoning and multiple languages.
It is fully open-source and was released by *HuggingFaceTB* in 2025.
The model was chosen due to its compatibility with LangChain's *HuggingFaceEndpoint* for API inference calls.
The embeddings used to construct the index are the **all-mpnet-base-v2** sentence embeddings.
Further information on the model & embeddings can be found here:
* [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
* [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

## RAG
The complete RAG setup is implemented with LangChain and consits of multiple parts.
1. **Index**: The index is created with the [FAISS](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html) vector store and is either built from scratch or loaded (if loal index file exists)
To construct the index, LangChain's *DirectoryLoader* in combination with *PyPDFLoader* (NOTE: the model only works with PDF files)
2. **Retriever**: The RAG retriever is constructed from the vector store. Additionally, a retriever tool is created to help the model decide whether to use the retriever for inference.
3. **Inference**: For inference, LangChain's [ChatHuggingFace](https://docs.langchain.com/oss/python/integrations/chat/huggingface#chathuggingface) chat model was used to be able to connect the LLM with the HF API
4. **Logging**: Basic logging is supported by the *logging* library

## Text-to-Speech Support
A TTS feature was implemented for audio support. The LLM output field in the Streamlit app now has a button to play German audio for the given text without saving it as a WAV file. The audio is generated for every individual response in the Q&A loop. The functionality was implemented with [PiperTTS](https://github.com/OHF-Voice/piper1-gpl).

## Docker
The app can be used with Docker as a Dockerfile is now provided. To build the container, run *docker build -t your-image-name .* in the command line. To use the app in a container, both the port and HF_API_TOKEN must be provided when running it. There are two options to do this:
1. Pass the token at docker run: *docker run -p 8501:8501 -e HUGGINGFACEHUB_API_TOKEN="your_api_token" your-image-name* (Here, the port is just an example and the token is exposed in the terminal)
2. Create a .env file and put the HF token there with the following pattern: *HUGGINGFACEHUB_API_TOKEN=your_api_token*. Te docker command is *docker run -p 8501:8501 --env-file .env your-image-name*

### Problems
This setup only works with LLMs compatible with the HF Chat methods. The model used in this project supports the task *conversational*. Hence, if models are used that don't support this task (e.g., text-generation only), the following error might occurr: **ValueError: Model X is not supported for task conversational. Supported task: text-generation.**
