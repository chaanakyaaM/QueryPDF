# 📘 LocalChat
**LocalChat** is a privacy-focused RAG based offline tool that lets you interact with PDF documents using natural language, completely offline. It extracts text from a PDF, breaks it into meaningful chunks, generates semantic embeddings, and stores them in a local vector database (ChromaDB). You can then ask questions, and it will retrieve relevant context and generate accurate, grounded answers using the lightweight gemma3:1b language model via Ollama.



# Requirements
- Python 3.8+

- Install dependencies:

```python
pip install pypdf transformers sentence-transformers langchain chromadb ollama
```
or 
```python
pip install -r requirements.txt
```
- Download the gemma3:1b (815MB) model in ollama 

- Ensure it by running ```ollama run gemma3:1b``` in the terminal

# 📂 Project Structure
```
project/
├── sample.pdf          # Your input PDF file
├── app.py              # Main script
└── README.md           
```

# ⚙️ How It Works
- PDF Extraction: Extracts text from pages 16–25 of sample.pdf.

- Tokenization & Chunking: Splits the text into ~450-token chunks with overlap.

- Embedding: Encodes each chunk into a vector using intfloat/e5-small-v2.

- Storage: Saves vectors and text chunks into a local ChromaDB.

- Chat Interface: Uses gemma3:1b to answer your questions based on relevant chunks.

# ▶️ Usage
Place your sample.pdf in the root directory.
Run the script:
```
python app.py
```
Ask your questions interactively:
```
Chat with PDF: What is a PDF?
💬 Response:
[Model-generated answer]
```
- Type ```/exit``` to quit the chat.

# ❗ Notes
- Ensure Ollama is using ```gemma3:1b``` model.

- Modify the ```reader.pages[15:25]``` line to adjust the page range as needed.

- Replace ```sample.pdf``` with your own file name.


# 🛠️ Tech Used

| Technology                                                 | Purpose                                                        |
| ---------------------------------------------------------- | -------------------------------------------------------------- |
| **[pypdf](https://pypi.org/project/pypdf/)**               | Extract text from PDF files                                    |
| **[transformers](https://huggingface.co/transformers/)**   | Tokenization using pretrained models                           |
| **[langchain](https://github.com/langchain-ai/langchain)** | Text chunking with recursive splitter                          |
| **[sentence-transformers](https://www.sbert.net/)**        | Generate dense vector embeddings                               |
| **[ChromaDB](https://www.trychroma.com/)**                 | Local vector database for storing and querying embeddings      |
| **[Ollama](https://ollama.com/)**                          | Run local LLMs like `gemma3:1b` for chat-style responses       |
