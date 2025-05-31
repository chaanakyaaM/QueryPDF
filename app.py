import os
import sys
import uuid
import json
import atexit
from yaspin import yaspin
from ollama import chat
from colorama import Fore, init
from pypdf import PdfReader
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize terminal formatting
init(autoreset=True)

def graceful_exit(message=None, exit_code=1):
    if message:
        print(Fore.RED + message)
    sys.exit(exit_code)

# === Configuration ===
def load_config(config_file="config.json"):
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        graceful_exit(f"Configuration file '{config_file}' not found.")

config = load_config()
embedding_model = config["embedding_model"]
ollama_model = config["ollama_model"]
chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]

# === ChromaDB Setup ===
try:
    client = chromadb.Client()
    collection = client.get_or_create_collection("my_chunks")
except Exception as e:
    graceful_exit(f"Error connecting to ChromaDB: {e}")

def cleanup():
    try:
        client.delete_collection("my_chunks")
        print(Fore.RED + "• Cleaned up ChromaDB collection.")
    except Exception as e:
        graceful_exit(f'Cleanup failed: {e}')
atexit.register(cleanup)

# === PDF Loading and Text Extraction ===
def extract_text_from_pdf():
    try:
        path = input("Paste the PDF path: ").strip()
        reader = PdfReader(path)
        print(Fore.GREEN + "• PDF loaded.")
    except FileNotFoundError:
        graceful_exit("File not found. Check the path.")

    try:
        range_input = input("Enter the page range (e.g., 15-25): ")
        start, end = map(int, range_input.split('-'))
        pages = reader.pages[start:end]
        print(Fore.GREEN + f"• Extracted pages {start}–{end}.")
        return '\n'.join([page.extract_text() or "" for page in pages])
    except Exception as e:
        graceful_exit(f"Error extracting text: {e}")

# === Text Chunking ===
def split_text(corpus, tokenizer):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(tokenizer.encode(text, truncation=True, max_length=tokenizer.model_max_length))
        )
        return splitter.split_text(corpus)
    except Exception as e:
        graceful_exit(f"Error splitting text: {e}")

# === Embedding Generation ===
def generate_embeddings(chunks):
    try:
        print(Fore.YELLOW + "• Generating embeddings...")
        model = SentenceTransformer(embedding_model)
        embeddings = model.encode(
            [f"passage: {chunk}" for chunk in chunks],
            show_progress_bar=True
        )
        print(Fore.GREEN + "• Embeddings generated.")
        return embeddings
    except Exception as e:
        graceful_exit(f"Error generating embeddings: {e}")

# === Store Embeddings ===
def store_chunks(chunks, embeddings):
    try:
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[str(uuid.uuid4()) for _ in chunks]
        )
        print(Fore.GREEN + "• Chunks stored in ChromaDB.")
    except Exception as e:
        graceful_exit(f"Error storing chunks: {e}")

# === Ask Question ===
def ask_ollama(query):
    print(Fore.YELLOW + "\n• Question:\n", query)
    print(Fore.GREEN + "\n• Response:")

    try:
        with yaspin(text="Generating response...", color="cyan") as spinner:
            results = collection.query(query_texts=[f"query: {query}"], n_results=3)
            context = "\n\n".join(results["documents"][0])

            prompt = (
                f"Using the provided context, answer the question clearly and directly.\n"
                f"Context: {context}\nQuestion: {query}\nRespond in plain text only."
            )

            response = chat(model=ollama_model, messages=[{"role": "user", "content": prompt}], stream=True)

            for idx, chunk in enumerate(response):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    if idx == 0:
                        spinner.ok("✔")
                    print(content, end="", flush=True)

            print()

    except Exception as e:
        graceful_exit(f"Error during chat: {e}")

# === Main ===
def main():
    corpus_data = extract_text_from_pdf()

    try:
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    except Exception as e:
        graceful_exit(f"Failed to load tokenizer: {e}")

    chunks = split_text(corpus_data, tokenizer)
    print(Fore.GREEN + f"• Split into {len(chunks)} chunks.")

    embeddings = generate_embeddings(chunks)
    store_chunks(chunks, embeddings)

    while True:
        query = input("\nChat with PDF : ")
        if query.lower() == '/exit':
            print(Fore.RED + "• Exiting.")
            break
        ask_ollama(query)

if __name__ == "__main__":
    main()
