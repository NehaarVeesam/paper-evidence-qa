# Paper RAG

Grounded question answering over a **research PDF you provide locally**, with **page-level citations** and answers from a lightweight local LLM.

> **PaperEvidenceQA:** retrieve relevant passages, answer only from that context, and cite pages—or abstain when the paper does not support an answer.

Users supply their own PDF. This repository does **not** redistribute copyrighted papers.

---

## What it does

1. **Extract** text from a PDF (per page, with page numbers)
2. **Chunk** text with overlap and metadata (`page`, `source`, `chunk_id`)
3. **Embed** chunks into a persistent ChromaDB index
4. **Retrieve** the top-k most similar passages for a question
5. **Generate** an answer with **Ollama** (`qwen2.5:3b`), grounded in retrieved context only

---

## Stack

| Component | Choice |
|-----------|--------|
| PDF parsing | PyMuPDF (`pymupdf` via conda-forge) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (`.chroma/` on disk) |
| LLM | Ollama — `qwen2.5:3b` |

---

## How it works

```
data/paper.pdf
      │
      ▼
 load_pdf.py  ──►  chunk.py  ──►  index.py  (embed + store in Chroma)
                                      │
                                      ▼
                              .chroma/  (local index)

User question
      │
      ▼
 retrieve.py  ──►  top-k chunks + page metadata
      │
      ▼
 generate.py  ──►  Ollama (qwen2.5:3b)  ──►  answer + page citations
```

---

## Project layout

```
project_folder/
├── README.md
├── environment.yml       # full env export (optional; large — see Python setup)
├── .env.example
├── .env                  # your local LLM settings (create from .env.example)
├── data/
│   └── paper.pdf         # your PDF (not committed to git)
├── .chroma/              # vector index (created by index command)
└── src/
    ├── config.py
    ├── load_pdf.py
    ├── chunk.py
    ├── index.py
    ├── retrieve.py
    ├── generate.py
    └── cli.py
```

---

## Full setup guide

### 1. Ollama setup (LLM server)

Ollama runs the **Qwen** model locally. Your Python code talks to it over HTTP (`http://127.0.0.1:11434`).

#### Install Ollama

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS / Windows:** download from [https://ollama.com/download](https://ollama.com/download)

Check install:

```bash
ollama --version
```

#### Pull the model (one-time download ~2 GB)

```bash
ollama pull qwen2.5:3b
```

Wait until you see `success`.

#### Start the Ollama server

Ollama usually runs automatically after install. To start it manually:

```bash
ollama serve
```

**Expected:** server listens on `http://127.0.0.1:11434`

**If you see `address already in use`:** Ollama is **already running**. You do **not** need another `ollama serve`. Continue to the verify step.

> **Tip:** Keep `ollama serve` running in a **separate terminal** only if your system does not start Ollama as a background service.

#### Verify Ollama is working

**Check API:**

```bash
curl http://127.0.0.1:11434/api/tags
```

You should see JSON listing `qwen2.5:3b`.

**Quick chat test:**

```bash
ollama run qwen2.5:3b "Say hello in one sentence."
```

First response may take 30–60 seconds while the model loads into memory.

---

### 2. Python setup (Conda)

#### Recommended: minimal manual install

Best for beginners and machines **without** a large GPU:

```bash
cd project_folder

conda create -n paper-rag python=3.11 -y
conda activate paper-rag

# PDF library — use conda-forge (avoids native library errors)
conda install -c conda-forge pymupdf -y

# RAG + CLI dependencies
pip install chromadb sentence-transformers python-dotenv typer openai
```

#### Optional: use `environment.yml`

```bash
conda env create -f environment.yml
conda activate paper-rag
```

> **Warning:** `environment.yml` is a full export of one machine’s env and may include heavy GPU packages (PyTorch, vllm). It can be **slow or fail** on laptops. Prefer the **manual install** above unless you know you need the full export.

Verify Python packages:

```bash
conda activate paper-rag
python -c "import fitz, chromadb, sentence_transformers; print('OK')"
```

---

### 3. Project configuration

From the project root (`project_folder/`):

```bash
# LLM settings (defaults work with local Ollama)
cp .env.example .env

# Folder for your PDF
mkdir -p data

# Copy your research PDF (must be text-selectable, not scan-only)
cp /path/to/your/paper.pdf data/paper.pdf
```

**Test that the PDF has extractable text:**

```bash
cd src
python load_pdf.py ../data/paper.pdf
```

You should see `Number of pages: ...` and readable text from page 1.

---

### 4. Environment variables (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | Ollama OpenAI-compatible API |
| `LLM_MODEL_NAME` | `qwen2.5:3b` | Model name from `ollama pull` |
| `LLM_API_KEY` | `ollama` | Placeholder; Ollama ignores it locally |

Only change these if Ollama runs on another host/port or you use a different model.

---

## Usage

Open a terminal, activate conda, go to `src/`:

```bash
conda activate paper-rag
cd project_folder/src
```

**Before `ask`:** ensure Ollama is running (see [verify step](#verify-ollama-is-working)).

### Step A — Index the paper (run once, or after PDF/config changes)

```bash
python cli.py index
```

Or with an explicit PDF path:

```bash
python cli.py index ../data/paper.pdf
```

**Expected output:** `Indexed N chunks into Chroma Collection`

**First run notes:**
- Downloads the embedding model from Hugging Face (may take a few minutes)
- Creates `.chroma/` in the project root

### Step B — Ask questions

```bash
python cli.py ask "What is the main contribution of this paper?"
python cli.py ask "What datasets are used for training?"
python cli.py ask "What are the stated limitations?"
```

Always wrap the question in **quotes**.

### Example output

**Question:** What are the datasets used?

**Answer:** Bullet list grounded in retrieved chunks, with page citations (e.g. datasets on pages 65–66).

---

## Configuration (`src/config.py`)

| Setting | Default | When to change |
|---------|---------|----------------|
| `PDF_PATH` | `data/paper.pdf` | Different default PDF location |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Dense two-column papers |
| `TOP_K` | `5` | More context per question |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Stronger embeddings (re-index required) |

After changing chunk or embedding settings, run **`python cli.py index`** again.

---

## Re-index when

- You replace `data/paper.pdf`
- You change `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBEDDING_MODEL_NAME`

---

## Limitations

- **Text-based PDFs only** — scanned/image-only papers need OCR (not included)
- **No figure or chart understanding** — only extracted text is indexed
- **Single paper** — one Chroma collection
- **Retrieval is approximate** — the best passage may rank 2nd–4th; top-k context still feeds the LLM

---

## Troubleshooting

### Ollama

| Issue | Fix |
|-------|-----|
| `Connection refused` on port 11434 | Run `ollama serve` or install/start Ollama service |
| `address already in use` on `ollama serve` | Ollama already running — **ignore**, continue |
| Model not found | `ollama pull qwen2.5:3b` |
| Very slow first answer | Normal — model loads once; wait 30–60s |
| `ollama: command not found` | Reinstall from [ollama.com/download](https://ollama.com/download) |

### Python / RAG

| Issue | Fix |
|-------|-----|
| `libtiff` / `jpeg` error importing PyMuPDF | `pip uninstall pymupdf -y` then `conda install -c conda-forge pymupdf libjpeg-turbo libtiff -y`. Ensure `load_pdf` is imported **before** `chromadb` in `index.py` |
| `ChromaDB directory not found` | Run `python cli.py index` first |
| Empty PDF text | PDF may be scan-only; use a text-selectable export |
| `Missing argument PDF_PATH` | Use `python cli.py index ../data/paper.pdf` |
| Slow first `index` | Embedding model download; later runs are faster |
| HuggingFace rate limit warning | Optional: set `HF_TOKEN` in env for faster downloads |

### End-to-end smoke test

```bash
# Terminal 1 — only if Ollama is not already running
ollama serve

# Terminal 2
conda activate paper-rag
cd project_folder/src
python cli.py index ../data/paper.pdf
python cli.py ask "What is the main contribution?"
```

---

## Development (debug individual steps)

```bash
cd src
python load_pdf.py ../data/paper.pdf
python chunk.py ../data/paper.pdf
python index.py
python retrieve.py "What is the main contribution?"
python generate.py "What datasets are used?"
```

---

## License

Add your license here (e.g. MIT). PDF content remains the property of its authors and publisher.
