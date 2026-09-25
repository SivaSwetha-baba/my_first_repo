Zepto GenAI RAG Service
A small offline-first GenAI service for Zepto policies using Sentence Transformers, ChromaDB, LangGraph, Pydantic, and FastAPI.
Graded baseline
The graded path is fully offline:
```bash
# Leave MOCK_LLM unset, or use:
set MOCK_LLM=1
```
No LLM provider is called in this mode. The keyword router, local embeddings, ChromaDB retrieval, deterministic mock answers, and Pydantic response validation are all local.
Project structure
```text
.
├── docs/
│   ├── doc_01.txt ... doc_08.txt
├── ingest.py
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```
Setup
Python 3.11+ is recommended.
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python ingest.py
```
The first `SentenceTransformer("all-MiniLM-L6-v2")` use may download the open-source model. After the model is present locally, the service itself makes no LLM-provider call in mock mode.
Architecture: ingestion → embedding → retrieval → generation
Ingestion / chunking — `ingest.py`
`load_documents()` loads all eight `docs/doc_*.txt` files.
Because each source document is short enough, each document is stored as one chunk.
Each chunk gets an ID such as `doc_01_chunk_01` and metadata containing the source document.
Embedding — `ingest.py` + `sentence-transformers`
`SentenceTransformer("all-MiniLM-L6-v2")` converts all eight chunks into vectors.
Embeddings are normalized before indexing.
Vector storage — ChromaDB
`ingest.py` creates a persistent ChromaDB collection named `zepto_policies`.
The collection uses cosine distance.
The local database is stored in `./chroma_db`.
Intent routing — `main.py` / LangGraph
`build_graph()` creates a `StateGraph` with the three required nodes:
`classify_intent`, `retrieve_and_answer`, and `direct_answer`.
A conditional edge after `classify_intent` routes policy questions to retrieval or general questions directly to `direct_answer`.
Retrieval — `retrieve_and_answer`
The query is embedded with the same MiniLM model.
ChromaDB returns the top 3 chunks by cosine similarity.
Retrieval is real in both mock and real-LLM modes.
Generation — `retrieve_and_answer` / `direct_answer`
In mock mode, `retrieve_and_answer` returns:
`Based on the retrieved context: {top_chunk_snippet}`
and `direct_answer` returns the fixed general-question message.
In `MOCK_LLM=0`, these nodes call the optional Groq backend. The structured prompt includes role, context, task, format, length, a negative grounding constraint, and a few-shot example.
MOCK_LLM branching
The toggle affects generation/classification, not vector retrieval:
`MOCK_LLM` unset or `1`: keyword intent heuristic + real local embedding/ChromaDB retrieval + deterministic canned generation. No LLM API call.
`MOCK_LLM=0`: LLM-based intent classification and LLM answer generation are enabled. Retrieval remains local and unchanged.
Structured prompt
`main.py` contains `STRUCTURED_PROMPT` with the required role/context/task/format/length skeleton, an explicit negative constraint, and a few-shot example. It is used by the optional real-LLM retrieval-answer path.
Pydantic output schema
The final API response is validated as:
```json
{
  "answer": "string",
  "sources": ["chunk_id"],
  "confidence": 0.0
}
```
In mock mode:
policy questions use the IDs of the three retrieved chunks;
general questions use `sources: []`;
`confidence` is deterministically `1.0`.
The optional real-LLM helper attempts validation up to three total times (initial attempt + two corrective retries). If validation still fails, it returns a clearly marked error response.
Run FastAPI locally
First build the index:
```bash
python ingest.py
```
Then:
```bash
# Windows PowerShell:
$env:MOCK_LLM="1"

uvicorn main:app --reload
```
Or leave `MOCK_LLM` unset; the code defaults to mock mode.
Example 1 — retrieval route
```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the delivery fee?\"}"
```
Representative raw JSON response after running the service:
```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149;","sources":["doc_01_chunk_01","doc_05_chunk_01","doc_04_chunk_01"],"confidence":1.0}
```
The exact order of the second and third retrieved chunks is determined by ChromaDB similarity.
Example 2 — direct route
```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the capital of France?\"}"
```
Raw JSON:
```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```
Docker
Build:
```bash
docker build -t zepto-genai .
```
Run:
```bash
docker run --rm -p 7860:7860 zepto-genai
```
Test:
```bash
curl -X POST http://127.0.0.1:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"How long do I have to report a damaged grocery item?\"}"
```
The Dockerfile runs `python ingest.py` during image build, so the image contains the local ChromaDB index and does not need an LLM API key for the graded baseline.
Optional real-LLM extension
The real-LLM path is not required for grading.
Installations already include `langchain-groq`. To use it:
```bash
# Windows PowerShell:
$env:MOCK_LLM="0"
$env:GROQ_API_KEY="your-key"

uvicorn main:app --reload
```
Never commit the API key. For a cloud deployment, use the platform's secret mechanism instead of hardcoding it.
Optional Hugging Face Spaces extension
A live deployment is not required for the graded submission. If used, deploy the same Dockerfile to a free Community CPU Space and store any API key as a Space secret. Do not commit secrets.
