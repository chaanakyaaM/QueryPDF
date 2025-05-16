# QueryPDF

**QueryPDF** is a privacy-focused, terminal-based RAG tool that allows you to interact with PDF documents using natural language, completely offline. It extracts text from user-specified pages, chunks it intelligently, embeds the content, and stores it in a local vector database. Then, using an LLM served by Ollama, it provides accurate, grounded responses to your queries — all without needing an internet connection.

## ✅ Requirements

* Python 3.8+
* Minimum 4GB RAM
* Ollama installed

## 📦 Installation

Create a virtual environment using:

```bash
python -m venv <environment-name>
```
Activate the virtual environment:

```bash
./<environment-name>/scripts/activate
```
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
├── config.json         # Configuration file
├── app.py              # Main script
├── sample.pdf          # Example PDF for testing
├── requirements.txt    # Package dependencies
└── README.md           # Documentation
```

## ⚙️ Configuration

The application uses a `config.json` file with the following parameters:

```json
{
    "embedding_model": "intfloat/e5-small-v2",
    "ollama_model": "gemma3:1b",
    "chunk_size": 450,
    "chunk_overlap": 100
}
```

You can modify these settings to adjust:
- The embedding model used for semantic search (you can find embedding models on [hugging face](https://huggingface.co/)) .
- The Ollama LLM model used for response generation as per your requirements and resources (you can find ollama models on [Ollama github page](https://github.com/ollama/ollama)) .
- Make sure to install the required LLM model by running ```ollama run <model-name>``` in the terminal -- This installs the model locally .
- Chunk size and overlap for text splitting as per the model .

## ⚙️ How It Works

1. **PDF Extraction:** Extracts text from specific pages based on user input (you specify the page range during runtime).

2. **Tokenization & Chunking:** Uses the specified tokenizer to split the text into chunks of configurable size (default: 450 tokens) with customizable overlap (default: 100 tokens).

3. **Embeddings:** Creates embeddings for each chunk using SentenceTransformers (default: `intfloat/e5-small-v2`).

4. **Vector Storage:** Stores the embeddings and original text chunks in a local ChromaDB collection.

5. **Chat Interface:** Accepts user queries, retrieves the top 3 most relevant chunks, and feeds them into an Ollama-served LLM (default: `gemma3:1b`) for response generation.

## ▶️ Usage

1. Run the script:

```bash
python app.py
```

2. When prompted, paste the full path to your PDF file:

```
Paste the PDF path: /path/to/your/document.pdf
```

3. Enter the page range you want to analyze:

```
Enter the page range (e.g., 15-25): 10-20
```

4. The application will process the PDF, extracting text, generating embeddings, and storing them in ChromaDB.

5. Enter your questions when prompted:

```
Chat with PDF: What is the main topic discussed in the document?
```

6. To exit the application, type:

```
Chat with PDF: /exit
```

## 🔍 Features

- **Privacy**: All processing happens on your machine, no data leaves your system
- **Interactive Page Selection**: Specify which pages to analyze at runtime
- **Graceful Exit Handling**: Automatically cleans up ChromaDB collections on exit
- **Visual Progress Feedback**: Shows embedding generation progress
- **Context-Aware Responses**: Uses semantic search to find the most relevant chunks for your questions

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| [pypdf](https://pypi.org/project/pypdf/) | Extract text from PDF files |
| [transformers](https://huggingface.co/docs/transformers/index) | Tokenization using pretrained models |
| [langchain](https://python.langchain.com/docs/introduction/) | Text chunking with recursive character splitter |
| [sentence-transformers](https://www.sbert.net/) | Generate dense vector embeddings |
| [ChromaDB](https://www.trychroma.com/) | Store and query embeddings locally |
| [Ollama](https://ollama.com/) | Run open-source LLMs locally |
