from pypdf import PdfReader
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from ollama import chat
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

# --- Step 1: Extract text from PDF ---
try:
    # Replace the "sample.pdf" with your PDF file, and add that PDF file to the root directory
    reader = PdfReader("sample.pdf")  
    print(Fore.GREEN + "• PDF loaded successfully.")
except FileNotFoundError:
    print(Fore.RED + "• File not found. Please check the file path.")


pages = reader.pages[15:25]  # Adjust the range of pages as needed like [10:100]
corpus_data = '\n'.join([page.extract_text() or "" for page in pages])

print(Fore.GREEN + f"• Extracted text from pages 15 to 25.")

# --- Step 2: Tokenizer for chunking ---
try:
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
except Exception as e:
    print(Fore.RED + f'• Error loading tokenizer: {e}')

def tiktoken_len(text):
    return len(tokenizer.encode(text, truncation=True, max_length=tokenizer.model_max_length))

# --- Step 3: Split into chunks ---
try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=100, length_function=tiktoken_len)
    chunks = splitter.split_text(corpus_data)
    print(Fore.GREEN + f"• Text split into {len(chunks)} chunks.")
except Exception as e:
    print(Fore.RED + f'• Error occurred while splitting text: {e}')


# --- Step 4: Generate embeddings ---
try:
    print(Fore.YELLOW + "• Generating embeddings...")
    model = SentenceTransformer("intfloat/e5-small-v2")
    embeddings = model.encode([f"passage: {chunk}" for chunk in chunks], show_progress_bar=True)
    print(Fore.GREEN + "• Embeddings generated.")
except Exception as e:
    print(Fore.RED + f'• Error occurred while generating embeddings: {e}')


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
    print(Fore.RED + f"• Error occurred while connecting to chromadb: {e}")

# --- Step 6: Chat with the PDF ---
def ask_ollama(query):
    results = collection.query(query_texts=[f"query: {query}"], n_results=3)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""Using the provided context, answer the following question as clearly and accurately as possible. Ensure your response directly addresses the question, without any additional or unnecessary information. Refer to the context for specific details, and make sure your answer is grounded in it.
                Context:
                {context}
                Question:
                {query}
                respond only in plain text (no markdown)."""
    
    response = chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}]
    )
    print(Fore.YELLOW + '\nQuestion:\n', query)
    print()
    print(Fore.GREEN + "Response:\n", response["message"]["content"])

# Main Loop for Chat with PDF
while True:
    query = input(Fore.CYAN + "Chat with PDF : ")
    if query.lower() == '/exit':
        print(Fore.RED + "Exiting program.")
        break
    else:
        ask_ollama(query)
