# Research Paper RAG

A small Python application that answers questions using a PDF you provide.
It retrieves relevant passages from a local vector database, sends those passages
and your question to a hosted Qwen model, and prints an answer with source labels
and PDF page references.

This is a learning project with explicit Python functions and HTTP requests.
It supports one PDF per collection and independent questions, without conversation
history. Answers and citations still need to be checked against the evidence.

## How it works

**Ingestion — run when preparing or replacing a document:**

```text
PDF → Extract page text → Split into chunks → Embed → Store in Chroma
```

**Question answering — run whenever you want to ask questions:**

```text
Question → Embed → Search Chroma → Build context → Qwen → Cited answer
```

The embedding model runs locally on CPU. Qwen generates answers through your
configured Chat Completions API. The paper does not become part of Qwen's trained
weights; selected passages are included in each request.

## Project structure

```text
.
├── .env.example
├── .gitignore
├── requirements.txt
├── docs/                   # Your PDFs; ignored by Git
├── data/                   # Generated vector database; ignored by Git
└── scripts/
    ├── config.py           # Environment settings and project paths
    ├── pdf_loader.py       # Extract page text and source metadata
    ├── chunking.py         # Split pages into overlapping passages
    ├── embeddings.py       # Load the local model and encode text
    ├── vector_store.py     # Persist, validate, and search records
    ├── retrieval.py        # Format context; retains the NumPy search example
    ├── llm_client.py       # Build messages and call Qwen
    ├── ingest.py           # Prepare a PDF for searching
    ├── ask.py              # Interactive question-answering loop
    └── inspect_store.py    # Inspect a stored record
```

## Setup

Run these commands from the project root. Conda must already be installed.

```bash
conda create --name paper-rag python=3.11 pip
conda activate paper-rag
python -m pip install -r requirements.txt
```

If the environment already exists, activate it and install the requirements.
The requirements pin the direct dependencies used in the working development
environment; they are not a full lockfile of every transitive dependency.

Create your local configuration if you do not already have a `.env`:

```bash
cp .env.example .env
```

Set your server's connection details in `.env`:

```dotenv
QWEN_BASE_URL=https://your-qwen-host.example/v1
QWEN_API_KEY=your-api-key
QWEN_MODEL=your-served-model-name
PDF_PATH=docs/document.pdf
```

The URL normally ends in `/v1`; a full `/chat/completions` URL also works. The
model name must match a model served by that endpoint. Leave the key blank only
if your server does not require authentication.

Place your own PDF at `docs/document.pdf`, or change `PDF_PATH`. Prefer a PDF with
selectable text. Scanned pages require OCR, which this project does not implement.
Source documents and prebuilt database files are not distributed with the code.

The first embedding-model load downloads its files, so internet access is needed
then. Subsequent loads can reuse the local model cache.

## Run

### 1. Ingest the PDF

```bash
python scripts/ingest.py
```

This extracts text, builds chunks, checks their token lengths, generates vectors,
and stores text, metadata, and vectors in Chroma. Blank extracted pages are skipped.

### 2. Ask questions

```bash
python scripts/ask.py
```

Example questions:

- What problem does the paper address?
- Which datasets are used for evaluation?
- What limitations do the authors discuss?

Type `/exit` to quit. Each question is independent; follow-up questions must include
their own context. Passage embeddings are reused from disk. Only the question is
embedded during each query.

The terminal shows the retrieved sources, their PDF page numbers and similarity
scores, the generated answer, and token usage when the server supplies it.
`[Source 1]` refers to the first retrieved source for that question.

### 3. Inspect stored data (optional)

```bash
python scripts/inspect_store.py
```

This prints the record count, one text preview, metadata, and vector dimension.
The preview contains document content; avoid sharing it if your document is private.

## Configuration

Settings live in `.env` or shell variables. Shell variables take precedence.
Relative paths resolve from the project root. Restart a script after changing
settings.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PDF_PATH` | `docs/document.pdf` | PDF to ingest |
| `VECTOR_DB_PATH` | `data/chromadb` | Local database directory |
| `COLLECTION_NAME` | `document_data` | Collection to store and search |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Model shared by passages and questions |
| `CHUNK_SIZE` | `400` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `50` | Characters shared by neighboring chunks on the same page |
| `TOP_K` | `5` | Number of passages retrieved per question |
| `QWEN_BASE_URL` | Required | Hosted API address |
| `QWEN_API_KEY` | Blank | Server credential |
| `QWEN_MODEL` | Required | Served chat-model name |
| `QWEN_MAX_TOKENS` | `1024` | Maximum generated tokens, potentially including reasoning |
| `QWEN_TEMPERATURE` | `0.2` | Sampling randomness |
| `QWEN_TIMEOUT` | `120` | HTTP connection/read timeout in seconds |
| `QWEN_ENABLE_THINKING` | Blank | Optional Qwen/vLLM switch; `true`/`false`, if supported |

Blank settings use defaults. The default embedding model produces 384-dimensional
vectors and accepts up to 256 tokens. Chunks that exceed the active model's input
limit cause an error rather than being silently truncated.

Increasing `TOP_K` supplies more passages, which can include irrelevant text.
Increasing `QWEN_MAX_TOKENS` permits longer output but does not request more detail
or improve evidence. The system prompt in `llm_client.py` controls answer guidance.

Legacy variables such as `QWEN_JSON_MODE`, `QWEN_TOP_P`, and `QWEN_MAX_MODEL_LEN`
are not used by this application. Existing `.env` files are left unchanged, so
review their values when upgrading from earlier examples.

## Replacing a document or changing embeddings

The collection records the source file's hash, filename, processing settings,
and whether ingestion completed. Querying checks the saved processing settings.
Changing a PDF on disk does not update the stored passages automatically.

To replace the document or change the model/chunk settings:

```bash
python scripts/ingest.py --rebuild
```

**This deletes and recreates the selected collection after extraction and embedding
succeed.** It does not delete your PDF or other collections. The replacement is
not an atomic transaction; if writing is interrupted, rerun ingestion to complete
it. Alternatively, choose a new `COLLECTION_NAME` to preserve the old collection.

Collections created by earlier versions lack these checks and require a one-time
rebuild. Keep the PDF available for that rebuild. Re-ingesting identical inputs
uses stable IDs to update records rather than duplicate them.

## Troubleshooting

| Problem | Action |
| --- | --- |
| PDF not found | Check `PDF_PATH` and supply your document |
| Missing or empty collection | Run ingestion before asking questions |
| Stored configuration mismatch | Rebuild with the intended PDF or choose a new collection |
| Ingestion incomplete | Run ingestion again |
| Chunk exceeds token limit | Reduce chunk size and rebuild |
| HTTP 401/403 | Verify credentials and access |
| HTTP 404 | Verify API URL and served model name |
| HTTP 400 | Check context/output limits and optional thinking-field support |
| Timeout | Verify server availability; reduce output budget or increase timeout |
| Short or unsupported answer | Inspect retrieved passages before increasing output length |
| `finish_reason=length` | Output may be incomplete; inspect the token budget and server limits |

Similarity scores are ranking signals, not probabilities of correctness. A top-k
search always returns the closest available passages, even for unrelated questions.
Prompts encourage abstention and citations but do not guarantee factual answers.

## Privacy and Git

- `.env`, `docs/`, `data/`, PDFs, logs, and Python caches are ignored by Git.
- The local database stores actual passage text as well as vectors and metadata.
  Removing a PDF alone does not remove its text from the database.
- Retrieved passages and your questions are sent to the configured Qwen endpoint.
  Local embeddings do not make the entire workflow local.
- Ignore rules do not remove files already tracked or present in commit history.

Before committing, inspect your staged files from your Git repository:

```bash
git status --short
git diff --cached --name-only
git diff --cached
```

## Current scope

This version uses character-based chunks, local dense-vector search, and a hosted
LLM. It does not implement OCR, multi-document ingestion, reranking, conversational
memory, or automated answer evaluation. Those can be added after measuring this
baseline on questions with known supporting evidence.
