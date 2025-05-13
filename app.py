from pypdf import PdfReader
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from ollama import chat


# --- Step 1: Extract text from PDF ---
try:
    # Replace the "sample.pdf" with your PDF file, and add that PDF file to the root directory
    reader = PdfReader("sample.pdf")  
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit(1)
pages = reader.pages[15:25]
corpus_data = '\n'.join([page.extract_text() or "" for page in pages])


# --- Step 2: Tokenizer for chunking ---
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")


def tiktoken_len(text):
    return len(tokenizer.encode(text, truncation=True, max_length=tokenizer.model_max_length))


# --- Step 3: Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=100, length_function=tiktoken_len)
chunks = splitter.split_text(corpus_data)


# --- Step 4: Generate embeddings ---
model = SentenceTransformer("intfloat/e5-small-v2")
embeddings = model.encode([f"passage: {chunk}" for chunk in chunks], show_progress_bar=True)


# --- Step 5: Store in ChromaDB ---
client = chromadb.Client()
collection = client.get_or_create_collection("my_chunks")
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(uuid.uuid4()) for _ in range(len(chunks))]
)


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
    print('\nQuestion:\n', query)
    print("💬 Response:\n", response["message"]["content"])


while True:
    query = input("Chat with PDF:")
    if query.lower() == '/exit':
        break
    else:
        ask_ollama(query)
