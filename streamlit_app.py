import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# --- CONFIGURATION ---
# Define paths and model names using the new Qwen3 model
CORPUS_PATH = "./corpus/"
VECTOR_STORE_PATH = "vectorstore/db_faiss"
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# IMPORTANT: After running tuning.py, you will have a folder named 'qwen3-4b-samsung-support-adapter'.
# Update this path to point to your saved LoRA adapter.
ADAPTER_PATH = "./qwen3-4b-samsung-support-adapter"
# ADAPTER_PATH = "./results_qwen3/checkpoint-150"  # Adjust this path as needed

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_embedding_model():
    """Loads the BGE embedding model from Hugging Face."""
    st.write("Loading embedding model...")
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

def create_vector_store(embedding_model):
    """Creates a FAISS vector store from documents in the corpus folder."""
    if not os.path.exists(CORPUS_PATH) or not os.listdir(CORPUS_PATH):
        st.error(f"Corpus folder '{CORPUS_PATH}' is empty or does not exist. Please add your PDF manuals.")
        st.stop()
        
    st.write(f"Creating vector store from documents in '{CORPUS_PATH}'...")
    # Load documents
    pdf_files = [f for f in os.listdir(CORPUS_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        st.error(f"No PDF files found in '{CORPUS_PATH}'. Please add your manuals.")
        st.stop()

    all_docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(CORPUS_PATH, pdf))
        all_docs.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(all_docs)

    if not text_chunks:
        st.error("Could not split the documents into text chunks. Check the PDF content.")
        st.stop()

    # Create and save the vector store
    vector_store = FAISS.from_documents(documents=text_chunks, embedding=embedding_model)
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    st.success(f"Vector store created and saved at '{VECTOR_STORE_PATH}'.")
    return vector_store

@st.cache_resource
def load_llm_and_adapter(_tokenizer):
    """Loads the base LLM and merges the fine-tuned LoRA adapter."""
    st.write("Loading base model and fine-tuned adapter...")

    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # Automatically use available GPUs
        trust_remote_code=True,
    )

    # Load and merge the LoRA adapter
    if not os.path.exists(ADAPTER_PATH):
        st.error(f"Adapter not found at '{ADAPTER_PATH}'. Please run tuning.py first.")
        st.stop()
        
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    st.success("Model and adapter loaded successfully.")
    return model

@st.cache_resource
def load_tokenizer():
    """Loads the tokenizer for the specified model."""
    st.write("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token
    return tokenizer

def get_response_from_llm(query, vector_store, model, tokenizer):
    """Performs RAG to get a response from the LLM."""
    st.write("Searching for relevant context...")
    # Retrieve relevant context from the vector store
    context_docs = vector_store.similarity_search(query, k=4)
    retrieved_context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])

    # Define the system prompt and user message for the Qwen chat template
    system_prompt = """You are an expert technical support assistant for Samsung S25 Ultra mobile phone. Your name is SamAI.
You are friendly, patient, and your answers are clear and easy to follow.
Use only the provided context to answer the user's question.
Structure your answers with lists or step-by-step instructions if applicable.
If the context does not contain the answer, state that you don't have enough information from the provided documents and suggest asking another question.
Do not make up information."""
    
    user_message = f"""**CONTEXT:**
---
{retrieved_context}
---

**USER'S QUESTION:** {query}
"""
    
    # Create the chat messages list
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Apply the chat template to create the prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    st.write("Generating response...")
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Get the length of the prompt tokens
    prompt_token_length = inputs.input_ids.shape[1]

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
    
    # *** FIX: Slice the output to get only the newly generated tokens ***
    generated_tokens = outputs[0, prompt_token_length:]
    
    # Decode only the new tokens to get the clean response
    cleaned_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return cleaned_response, retrieved_context

# --- STREAMLIT UI ---

st.set_page_config(page_title="Samsung S25 Ultra AI Support", layout="wide")

st.title(" Samsung S25 Ultra AI Support Assistant (Qwen3-4B Edition)")
st.markdown("Your personal expert on how to get the most out of your Samsung S25 Ultra, powered by Qwen3 and RAG.")

# --- INITIALIZATION ---
embedding_model = load_embedding_model()
tokenizer = load_tokenizer()

# Check if vector store exists, otherwise create it
if os.path.exists(VECTOR_STORE_PATH):
    st.write(f"Loading existing vector store from '{VECTOR_STORE_PATH}'...")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    st.success("Vector store loaded.")
else:
    with st.spinner("First-time setup: Creating vector store from PDF documents. This may take a few minutes..."):
        vector_store = create_vector_store(embedding_model)

# Load the fine-tuned model
model = load_llm_and_adapter(tokenizer)


# --- INTERACTIVE CHAT ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your Samsung S25 Ultra today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("e.g., How do I set up a new user account?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response, context = get_response_from_llm(prompt, vector_store, model, tokenizer)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
            with st.expander("Show Retrieved Context"):
                st.info(context)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})