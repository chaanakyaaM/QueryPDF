import os
import sys
import uuid
import json
import atexit
import chromadb
from yaspin import yaspin 
from ollama import chat
from colorama import Fore, init
from pypdf import PdfReader
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize colorama
init(autoreset=True)

# Function to handle graceful exit
def graceful_exit(message=None, exit_code=1):
    if message:
        print(Fore.RED + f"{message}")
    sys.exit(exit_code)

# Connect to ChromaDB
try:
    client = chromadb.Client()
    collection = client.get_or_create_collection("my_chunks")
except Exception as e:
    graceful_exit(f"• Error occurred while connecting to ChromaDB: {e}")

    
# Ensure ChromaDB collection is deleted on exit/KeyboardInterrupt
def cleanup():
    try:
        client.delete_collection("my_chunks")
        print(Fore.RED + "• Collection 'my_chunks' deleted on exit.")
    except:
        pass  # Avoid raising further errors during exit
atexit.register(cleanup)


# Load configuration from JSON file
def load_config(config_file="config.json"):
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        graceful_exit(f"• Configuration file '{config_file}' not found.")

# Load config
config = load_config()
embedding_model = config["embedding_model"]
ollama_model = config["ollama_model"]
chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]

# --- Step 1: Extract text from PDF ---
try:
    pdf_path = input("Paste the PDF path: ").strip()
    reader = PdfReader(pdf_path)  
    print(Fore.GREEN + "• PDF loaded successfully.")

except FileNotFoundError:
    graceful_exit( "• File not found. Please check the file path.")

try:
    # Get the page range from the user
    page_range_input = input("Enter the page range (e.g., 15-25): ")
    start_page, end_page = map(int, page_range_input.split('-'))
    pages = reader.pages[start_page:end_page]  
    print(Fore.GREEN + f"• Extracted text from {start_page} to {end_page}.")
    corpus_data = '\n'.join([page.extract_text() or "" for page in pages])
except Exception as e:
    graceful_exit(f"• Error occurred while extracting text {e}")


# --- Step 2: Tokenizer for chunking ---
try:
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

except Exception as e:
    graceful_exit(f'• Error loading tokenizer: {e}')

def tiktoken_len(text):
    return len(tokenizer.encode(text, truncation=True, max_length=tokenizer.model_max_length))


# --- Step 3: Split into chunks ---
try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=tiktoken_len)
    chunks = splitter.split_text(corpus_data)
    print(Fore.GREEN + f"• Text split into {len(chunks)} chunks.")

except Exception as e:
    graceful_exit(f'• Error occurred while splitting text: {e}')


# --- Step 4: Generate embeddings ---
try:
    print(Fore.YELLOW + "• Generating embeddings...")
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode([f"passage: {chunk}" for chunk in chunks], show_progress_bar=True)
    print(Fore.GREEN + "• Embeddings generated.")

except Exception as e:
    graceful_exit( f'• Error occurred while generating embeddings: {e}')


# --- Step 5: Store in ChromaDB ---
try:
    client = chromadb.Client()
    collection = client.get_or_create_collection("my_chunks")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[str(uuid.uuid4()) for _ in range(len(chunks))]
    )
    print(Fore.GREEN + "• Data stored in ChromaDB.")

except Exception as e:
    graceful_exit(f"• Error occurred while adding the embeddings to ChromaDB: {e}")
    

# --- Step 6: Chat with the PDF ---
def ask_ollama(query):
    with yaspin(text="Generating response...", color="cyan") as spinner:
        try:
            results = collection.query(query_texts=[f"query: {query}"], n_results=3)
            context = "\n\n".join(results["documents"][0])

            prompt = f"""Using the provided context, answer the following question as clearly and accurately as possible. Ensure your response directly addresses the question, without any additional or unnecessary information. Refer to the context for specific details, and make sure your answer is grounded in it.
                        Context:
                        {context}
                        Question:
                        {query}
                        respond only in plain text (no markdown)."""

            response = chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}]
            )

            spinner.ok("✔")

        except Exception as e:
            spinner.fail("✖")
            graceful_exit(f"• Error during generating the response: {e}")

    print(Fore.YELLOW + '\nQuestion:\n', query)
    print()
    print(Fore.GREEN + "Response:\n", response["message"]["content"])


# Main Loop for Chat with PDF
try:
    while True:
        query = input("Chat with PDF : ")
        if query.lower() == '/exit':
            print(Fore.RED + "Exiting program.")
            break
        else:
            ask_ollama(query)

finally:
    cleanup() 