# QueryPDF

**QueryPDF** is a privacy-focused, terminal based RAG tool that allows you to interact with PDF documents using natural language, completely offline. It extracts text from specified pages, chunks it intelligently, embeds the content, and stores it in a local vector database. Then, using an LLM served by ollama, it provides accurate, grounded responses to your queries — all without needing an internet connection.

# ✅ Requirements

* Python 3.8+
* Minimum 4GB RAM

## 📦 Installation

Install the dependencies:

```bash
pip install pypdf transformers sentence-transformers langchain chromadb ollama colorama yaspin
```

Or using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
project/
├── sample.pdf   # Your input PDF file
├── app.py       # Main script
└── README.md
```

## ⚙️ How It Works

1. **PDF Extraction:** Extracts text from specific pages (default: page 16 only).

2. **Tokenization & Chunking:** Uses `intfloat/e5-small-v2` tokenizer to split the text into ~450-token chunks with 100-token overlaps.

3. **Embeddings:** Creates Embeddings for each and every chunk using SentenceTransformers (`e5-small-v2`).

4. **Vector Storage:** Stores the embeddings and original text chunks in a local ChromaDB collection.

5. **Chat Interface:** Accepts user queries, retrieves top 3 most relevant chunks, and feeds them into an Ollama-served LLM (`gemma3:1b`) for response generation.

## ▶️ Usage
1. Run the script:

```bash
python app.py
```
2. Paste the PDF destination path (including .pdf extension)
3. Enter your question when prompted:

```bash
Chat with PDF: What is a PDF?
💬 Response: [Model-generated answer]
```

4. To exit the application, type:

```bash
/exit
```

## 📝 Notes

* You can adjust the PDF page range in this line of the code:

```python
pages = reader.pages[15:16]  # Change to desired page range like [10:25]
```

* The default model used is `gemma3:1b`, but you can swap it for another supported Ollama model based on your requirements and resources. Find more at [Ollama GitHub](https://github.com/ollama/ollama).


## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| [pypdf](https://pypi.org/project/pypdf/) | Extract text from PDF files |
| [transformers](https://huggingface.co/docs/transformers/index) | Tokenization using pretrained models |
| [langchain](https://python.langchain.com/docs/introduction/) | Text chunking with recursive character splitter |
| [sentence-transformers](https://www.sbert.net/) | Generate dense vector embeddings |
| [ChromaDB](https://www.trychroma.com/) | Store and query embeddings locally |
| [Ollama](https://ollama.com/) | Run open-source LLMs locally |
