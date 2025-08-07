# AI Technical Support Assistant for Samsung S25 Ultra

This project is a fully functional AI-powered chatbot designed to act as an expert technical support assistant for the Samsung S25 Ultra mobile phone. It leverages a powerful combination of Retrieval-Augmented Generation (RAG) and model fine-tuning to provide accurate, helpful, and context-aware answers based on official documentation.

## Key Features

* **Interactive Chat Interface:** A user-friendly web interface built with Streamlit for easy interaction.
* **Fact-Grounded Responses:** Utilizes a RAG pipeline to retrieve relevant information from a knowledge base (PDF user manuals), ensuring answers are accurate and preventing hallucination.
* **Specialized Persona:** The base language model (`Qwen/Qwen3-4B-Instruct-2507`) has been fine-tuned using QLoRA to adopt the persona of a patient, clear, and friendly Samsung support agent.
* **Efficient & Local:** The entire system runs locally, leveraging a FAISS vector store for fast, in-memory similarity searches.

## Architecture: RAG + Fine-Tuning

This project demonstrates a state-of-the-art approach to building specialized AI assistants:

1.  **Fine-Tuning (The "Personality"):** The base Qwen3-4B model was fine-tuned on the `databricks-dolly-15k` dataset. This process doesn't teach the model new facts but rather adjusts its **style and persona**. The goal was to train it to communicate like a helpful support agent—structuring answers clearly, using step-by-step instructions, and maintaining a supportive tone. This is handled by `tuning.py`.
2.  **RAG (The "Knowledge"):** The RAG pipeline provides the model with factual, up-to-date information at inference time. When a user asks a question, the system:
    * Scans a vector database (built from Samsung user manuals) to find the most relevant text chunks.
    * Injects these chunks as context into a prompt for the fine-tuned model.
    * The model then generates an answer based *only* on the provided context, ensuring factual accuracy. This is handled by `app.py`.

## Tech Stack

* **LLM:** `Qwen/Qwen3-4B-Instruct-2507`
* **Fine-Tuning:** Hugging Face `transformers`, `peft` (QLoRA), `bitsandbytes`, `trl`
* **RAG & Vector Store:** `langchain`, `faiss-gpu`
* **Embeddings:** `BAAI/bge-base-en-v1.5`
* **UI:** `streamlit`

## How to Run This Project

### 1. Environment Setup

**Prerequisites:**

* Python 3.10+
* An NVIDIA GPU with CUDA 12.1+ installed
* Conda package manager

**Installation:**

```bash
# Clone the repository
git clone [https://github.com/Saivivekmulukuri/AI-Support-Assistant.git](https://github.com/Saivivekmulukuri/AI-Support-Assistant.git)
cd AI-Support-Assistant

# Create and activate a Conda environment
conda create -n qwen python=3.10
conda activate qwen

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install project dependencies
pip install -U transformers datasets peft trl bitsandbytes accelerate langchain langchain-community streamlit faiss-gpu pypdf tensorboard
```

### 2. Prepare Data

* Create a folder named `corpus` in the project's root directory.
* Download a Samsung S25 Ultra user manual as a PDF and place it inside the `corpus` folder.

### 3. Fine-Tune the Model (One-Time Step)

Run the tuning script to create the LoRA adapter. This will download the base Qwen model and the training dataset, which may take some time.

```bash
python finetuning_qwen.py
```

This will create a new folder named `qwen3-4b-samsung-support-adapter`, which contains the fine-tuned model weights.

### 4. Launch the Application

Run the Streamlit app. The first time you launch, it will create the FAISS vector store from your PDF document.

```bash
streamlit run streamlit_app.py
```

Open your web browser to the local URL provided by Streamlit to start chatting with your AI assistant.
