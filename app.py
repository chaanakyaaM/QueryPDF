# # import os
# # import sys
# # import uuid
# # import json
# # import atexit
# # import chromadb
# # from yaspin import yaspin 
# # from ollama import chat
# # from colorama import Fore, init
# # from pypdf import PdfReader
# # from transformers import AutoTokenizer
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from sentence_transformers import SentenceTransformer

# # # Initialize colorama
# # init(autoreset=True)

# # # Function to handle graceful exit
# # def graceful_exit(message=None, exit_code=1):
# #     if message:
# #         print(Fore.RED + f"{message}")
# #     sys.exit(exit_code)

# # # Connect to ChromaDB
# # try:
# #     client = chromadb.Client()
# #     collection = client.get_or_create_collection("my_chunks")
# # except Exception as e:
# #     graceful_exit(f"• Error occurred while connecting to ChromaDB: {e}")

    
# # # Ensure ChromaDB collection is deleted on exit/KeyboardInterrupt
# # def cleanup():
# #     try:
# #         client.delete_collection("my_chunks")
# #         print(Fore.RED + "• Collection 'my_chunks' deleted on exit.")
# #     except:
# #         pass  # Avoid raising further errors during exit
# # atexit.register(cleanup)


# # # Load configuration from JSON file
# # def load_config(config_file="config.json"):
# #     try:
# #         with open(config_file, 'r') as file:
# #             config = json.load(file)
# #         return config
# #     except FileNotFoundError:
# #         graceful_exit(f"• Configuration file '{config_file}' not found.")

# # # Load config
# # config = load_config()
# # embedding_model = config["embedding_model"]
# # ollama_model = config["ollama_model"]
# # chunk_size = config["chunk_size"]
# # chunk_overlap = config["chunk_overlap"]

# # # --- Step 1: Extract text from PDF ---
# # try:
# #     pdf_path = input("Paste the PDF path: ").strip()
# #     reader = PdfReader(pdf_path)  
# #     print(Fore.GREEN + "• PDF loaded successfully.")

# # except FileNotFoundError:
# #     graceful_exit( "• File not found. Please check the file path.")

# # try:
# #     # Get the page range from the user
# #     page_range_input = input("Enter the page range (e.g., 15-25): ")
# #     start_page, end_page = map(int, page_range_input.split('-'))
# #     pages = reader.pages[start_page:end_page]  
# #     print(Fore.GREEN + f"• Extracted text from {start_page} to {end_page}.")
# #     corpus_data = '\n'.join([page.extract_text() or "" for page in pages])
# # except Exception as e:
# #     graceful_exit(f"• Error occurred while extracting text {e}")


# # # --- Step 2: Tokenizer for chunking ---
# # try:
# #     tokenizer = AutoTokenizer.from_pretrained(embedding_model)

# # except Exception as e:
# #     graceful_exit(f'• Error loading tokenizer: {e}')

# # def tiktoken_len(text):
# #     return len(tokenizer.encode(text, truncation=True, max_length=tokenizer.model_max_length))


# # # --- Step 3: Split into chunks ---
# # try:
# #     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=tiktoken_len)
# #     chunks = splitter.split_text(corpus_data)
# #     print(Fore.GREEN + f"• Text split into {len(chunks)} chunks.")

# # except Exception as e:
# #     graceful_exit(f'• Error occurred while splitting text: {e}')


# # # --- Step 4: Generate embeddings ---
# # try:
# #     print(Fore.YELLOW + "• Generating embeddings...")
# #     model = SentenceTransformer(embedding_model)
# #     embeddings = model.encode([f"passage: {chunk}" for chunk in chunks], show_progress_bar=True)
# #     print(Fore.GREEN + "• Embeddings generated.")

# # except Exception as e:
# #     graceful_exit( f'• Error occurred while generating embeddings: {e}')


# # # --- Step 5: Store in ChromaDB ---
# # try:
# #     client = chromadb.Client()
# #     collection = client.get_or_create_collection("my_chunks")
# #     collection.add(
# #         documents=chunks,
# #         embeddings=embeddings.tolist(),
# #         ids=[str(uuid.uuid4()) for _ in range(len(chunks))]
# #     )
# #     print(Fore.GREEN + "• Data stored in ChromaDB.")

# # except Exception as e:
# #     graceful_exit(f"• Error occurred while adding the embeddings to ChromaDB: {e}")


# # # --- Step 6: Chat with PDF ---
# # def ask_ollama(query):
# #     print(Fore.YELLOW + '\n• Question:\n', query)
# #     print()
# #     print(Fore.GREEN + "• Response:")

# #     try:
# #         with yaspin(text="Generating response...", color="cyan") as spinner:
# #             results = collection.query(query_texts=[f"query: {query}"], n_results=3)
# #             context = "\n\n".join(results["documents"][0])

# #             prompt = f"""Using the provided context, answer the following question as clearly and accurately as possible. Ensure your response directly addresses the question, without any additional or unnecessary information. Refer to the context for specific details, and make sure your answer is grounded in it.
# #                         Context:{context} Question:{query} respond only in plain text (no markdown)."""

# #             response = chat(
# #                 model=ollama_model,
# #                 messages=[{"role": "user", "content": prompt}],
# #                 stream=True
# #             )

# #             first_chunk_printed = False
# #             for chunk in response:
# #                 content = chunk.get("message", {}).get("content", "")
# #                 if content:
# #                     if not first_chunk_printed:
# #                         spinner.ok("✔")
# #                         first_chunk_printed = True
# #                     print(content, end="", flush=True)

# #             if not first_chunk_printed:
# #                 spinner.fail("✖")
# #                 graceful_exit("• No content received from model.")

# #             print()  # newline after full response

# #     except Exception as e:
# #         graceful_exit(f"• Error while streaming response: {e}")



# # # Main Loop for Chat with PDF
# # try:
# #     while True:
# #         query = input("Chat with PDF : ")
# #         if query.lower() == '/exit':
# #             print(Fore.RED + "• Exiting program.")
# #             break
# #         else:
# #             ask_ollama(query)

# # finally:
# #     cleanup() 


# import os
# import sys
# import uuid
# import json
# import atexit
# import logging
# import chromadb
# from yaspin import yaspin 
# from ollama import chat
# from colorama import Fore, init
# from pypdf import PdfReader
# from transformers import AutoTokenizer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer

# # Initialize colorama
# init(autoreset=True)

# # Setup logger
# def setup_logger(log_file="app.log"):
#     logger = logging.getLogger("PDFChatLogger")
#     logger.setLevel(logging.DEBUG)

#     # Console handler
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
#     ch.setFormatter(ch_formatter)

#     # File handler
#     fh = logging.FileHandler(log_file)
#     fh.setLevel(logging.DEBUG)
#     fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     fh.setFormatter(fh_formatter)

#     logger.addHandler(ch)
#     logger.addHandler(fh)
#     return logger

# logger = setup_logger()

# # Function to handle graceful exit
# def graceful_exit(message=None, exit_code=1):
#     if message:
#         logger.error(message)
#         print(Fore.RED + message)
#     sys.exit(exit_code)

# # Connect to ChromaDB
# try:
#     client = chromadb.Client()
#     collection = client.get_or_create_collection("my_chunks")
# except Exception as e:
#     graceful_exit(f"Error occurred while connecting to ChromaDB: {e}")

# # Ensure ChromaDB collection is deleted on exit
# def cleanup():
#     try:
#         client.delete_collection("my_chunks")
#         print(Fore.RED + "• Collection 'my_chunks' deleted on exit.")
#     except Exception as e:
#         graceful_exit(f"Cleanup failed: {e}")

# atexit.register(cleanup)

# # Load configuration from JSON file
# def load_config(config_file="config.json"):
#     try:
#         with open(config_file, 'r') as file:
#             config = json.load(file)
#         return config
#     except FileNotFoundError:
#         graceful_exit(f"Configuration file '{config_file}' not found.")

# # Load config
# config = load_config()
# embedding_model = config["embedding_model"]
# ollama_model = config["ollama_model"]
# chunk_size = config["chunk_size"]
# chunk_overlap = config["chunk_overlap"]

# # Step 1: Extract text from PDF
# try:
#     pdf_path = input("Paste the PDF path: ").strip()
#     reader = PdfReader(pdf_path)
#     print(Fore.GREEN + "• PDF loaded successfully.")
# except FileNotFoundError:
#     graceful_exit("File not found. Please check the file path.")

# try:
#     page_range_input = input("Enter the page range (e.g., 15-25): ")
#     start_page, end_page = map(int, page_range_input.split('-'))
#     pages = reader.pages[start_page:end_page]
#     print(Fore.GREEN + f"• Extracted text from {start_page} to {end_page}.")
#     corpus_data = '\n'.join([page.extract_text() or "" for page in pages])
# except Exception as e:
#     graceful_exit(f"Error occurred while extracting text: {e}")

# # Step 2: Tokenizer for chunking
# try:
#     tokenizer = AutoTokenizer.from_pretrained(embedding_model)
# except Exception as e:
#     graceful_exit(f"Error loading tokenizer: {e}")

# def tiktoken_len(text):
#     return len(tokenizer.encode(text, truncation=True, max_length=tokenizer.model_max_length))

# # Step 3: Split into chunks
# # --- Step 3: Split into chunks ---
# try:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=tiktoken_len
#     )
#     chunks = splitter.split_text(corpus_data)
#     print(Fore.GREEN + f"• Text split into {len(chunks)} chunks.")

# except Exception as e:
#     logger.error(f"Error while splitting text: {e}")
#     graceful_exit(f'• Error occurred while splitting text: {e}')


# # --- Step 4: Generate embeddings ---
# try:
#     print(Fore.YELLOW + "• Generating embeddings...")
#     model = SentenceTransformer(embedding_model)
#     embeddings = model.encode(
#         [f"passage: {chunk}" for chunk in chunks],
#         show_progress_bar=True
#     )
#     print(Fore.GREEN + "• Embeddings generated.")

# except Exception as e:
#     logger.error(f"Error generating embeddings: {e}")
#     graceful_exit(f'• Error occurred while generating embeddings: {e}')


# # --- Step 5: Store in ChromaDB ---
# try:
#     client = chromadb.Client()
#     collection = client.get_or_create_collection("my_chunks")
#     collection.add(
#         documents=chunks,
#         embeddings=embeddings.tolist(),
#         ids=[str(uuid.uuid4()) for _ in range(len(chunks))]
#     )
#     print(Fore.GREEN + "• Data stored in ChromaDB.")

# except Exception as e:
#     logger.error(f"Error storing data in ChromaDB: {e}")
#     graceful_exit(f"• Error occurred while adding the embeddings to ChromaDB: {e}")


# # --- Step 6: Chat with PDF ---
# def ask_ollama(query):
#     print(Fore.YELLOW + '\n• Question:\n', query)
#     print()
#     print(Fore.GREEN + "• Response:")

#     try:
#         with yaspin(text="Generating response...", color="cyan") as spinner:
#             results = collection.query(query_texts=[f"query: {query}"], n_results=3)
#             context = "\n\n".join(results["documents"][0])
#             prompt = f"""Using the provided context, answer the following question as clearly and accurately as possible. Ensure your response directly addresses the question, without any additional or unnecessary information. Refer to the context for specific details, and make sure your answer is grounded in it.
#                         Context:{context} Question:{query} respond only in plain text (no markdown)."""

#             response = chat(
#                 model=ollama_model,
#                 messages=[{"role": "user", "content": prompt}],
#                 stream=True
#             )

#             first_chunk_printed = False
#             for chunk in response:
#                 content = chunk.get("message", {}).get("content", "")
#                 if content:
#                     if not first_chunk_printed:
#                         spinner.ok("✔")
#                         first_chunk_printed = True
#                     print(content, end="", flush=True)

#             if not first_chunk_printed:
#                 spinner.fail("✖")
#                 logger.error("No content received from Ollama model.")
#                 graceful_exit("• No content received from model.")

#             print()

#     except Exception as e:
#         logger.error(f"Error while streaming response: {e}")
#         graceful_exit(f"• Error while streaming response: {e}")


# # --- Main Loop for Chat with PDF ---
# try:
#     while True:
#         query = input("Chat with PDF : ")
#         if query.lower() == '/exit':
#             print(Fore.RED + "• Exiting program.")
#             break
#         else:
#             ask_ollama(query)

# finally:
#     cleanup()


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
