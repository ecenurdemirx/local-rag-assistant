# Local RAG Assistant with Microsoft Foundry Local

## Project Overview

A small, offline document Q&A assistant that follows the Summer School Foundry Local plan. It indexes local text, Markdown, and text-based PDF documents, retrieves the most relevant passages using embeddings, and asks an on-device chat model to answer using only that retrieved context.

All model inference runs locally. Internet access is required only for the initial SDK and model downloads; models are cached for offline use thereafter.

RAG has three stages: **retrieve** the most relevant passages, **augment** the model prompt with those passages, and **generate** an answer constrained by that context. Supplying relevant evidence reduces unsupported guessing, while deterministic source names let a user inspect where the answer came from.

## Architecture / RAG Flow

```text
documents/**/*.txt, documents/**/*.md, or documents/**/*.pdf
  -> word-safe chunking (approx. 800 chars, 120-char overlap)
  -> Foundry Local embedding model (qwen3-embedding-0.6b)
  -> SQLite (text, source, chunk index, embedding)
  -> query embedding + cosine similarity -> top K
  -> similarity threshold -> explicit answerability classification
  -> local model selects verbatim evidence passages
  -> independent answer verification (support + completeness)
  -> answer + deterministic source list
```

## Features

- Recursively loads UTF-8 `.txt` and `.md` files plus text-based `.pdf` files from a documents directory
- Word-safe, configurable chunking with overlap
- Batched local embedding calls via Foundry Local SDK
- Atomic SQLite reindexing (preserves old index on failure)
- Cosine-similarity retrieval for small document collections
- Strict grounded prompts, extractive answer validation, and explicit "insufficient information" responses
- Deterministic source citations taken from the index, not hallucinated
- Interactive CLI, retrieval evaluation scripts, and unit tests

## Requirements

- Python 3.11 or newer
- Compatible Foundry Local hardware on macOS (Apple Silicon), Linux, or Windows; this project was checked with Python 3.11 and SDK 1.2.4 on macOS.
- At least 8 GB RAM recommended for comfortable local model usage

## Foundry Local setup

Install and verify Foundry Local on macOS:

```bash
brew tap microsoft/foundrylocal
brew install foundrylocal

foundry --version
foundry model list
# Optional direct model check:
foundry chat qwen2.5-0.5b
```

## Installation

Create the Python 3.11 environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application uses the native SDK directly; a separately running HTTP service or cloud API key is not required. Keep the SDK version in `requirements.txt`: newer major versions may require API changes. For platform-specific installation, see the [Microsoft SDK instructions](https://github.com/microsoft/Foundry-Local/blob/main/sdk/python/README.md). On Windows, activate the environment using `.venv\Scripts\Activate.ps1` in PowerShell. The SDK does not require a cloud Foundry account for local usage.

## How to index documents

Place `.txt`, `.md`, and text-based `.pdf` documents under the `documents/` directory. Subfolders are preserved and used in source citations. PDF text is extracted page by page in document order and then passed through the same chunking pipeline as text and Markdown. Scanned/image-only PDFs are not OCR-processed. For teaching and experimentation, a focused set of ~5–10 short documents is recommended.

```bash
python main.py --reindex
```

`rag.db` is created automatically next to the Python files and is not required in the submission. The first run can also build the index automatically. Reindexing with no readable, non-empty documents fails clearly and preserves the previous index; it does not clear old data.

Reindex whenever you add, remove, or edit documents. A change to the embedding model also requires reindexing.

## How to run the application

```bash
python main.py
```

Interactive commands include: `reindex`, `exit`, and `quit`. If the database is empty but documents exist, the index is built automatically at startup.

## How to run tests

`evaluate.py` measures source retrieval only, not generated-answer correctness. Check answers interactively using the demo plan as well.

Unit tests are designed to run without downloading models and cover core functionality such as chunking, cosine similarity, atomic DB replacement, empty inputs, and insufficient-context handling:

```bash
python -m unittest discover -s tests -v
```

Evaluation cases are knowledge-base-specific, so `evaluate.py` does not assume that any bundled sample document exists. Supply a JSON cases file that matches your own indexed documents. The repository includes `evaluation_cases.example.json` only as demo data for the bundled five-document knowledge base:

```bash
python evaluate.py --cases evaluation_cases.example.json
```

To create evaluation data for your own knowledge base, copy the example and edit every question and expected source:

```bash
cp evaluation_cases.example.json evaluation_cases.json
python evaluate.py --cases evaluation_cases.json
```

The JSON format is:

```json
[
  {
    "question": "A question supported by your documents",
    "expected_source": "your_document.txt"
  },
  {
    "question": "An unsupported question",
    "expected_source": null
  }
]
```

`expected_source` must match the indexed relative filename. Use `null` when no source should score above the configured similarity threshold. A missing, malformed, empty, or structurally invalid cases file produces a clear command-line error instead of running unrelated demo assumptions.

## Example questions

```text
You: What is binary classification?

Assistant:
Binary classification has two possible classes.

Sources:
* classification_basics.txt

You: What is the capital of Japan?

Assistant:
I don't have enough information in the provided documents to answer this question.
```

## Configuration

Environment variables (defaults shown):

| Variable | Default | Purpose |
|---|---:|---|
| `RAG_DOCUMENTS_DIR` | `documents/` | Document root |
| `RAG_DATABASE_PATH` | `rag.db` | SQLite database file |
| `RAG_TOP_K` | `3` | Number of retrieved chunks to include |
| `RAG_SIMILARITY_THRESHOLD` | `0.55` | Minimum cosine similarity to accept a chunk |
| `RAG_CHUNK_SIZE` | `800` | Approximate chunk size (characters) |
| `RAG_CHUNK_OVERLAP` | `120` | Approximate chunk overlap (characters) |
| `RAG_EMBEDDING_MODEL` | `qwen3-embedding-0.6b` | Embedding model alias |
| `RAG_CHAT_MODEL` | `qwen2.5-0.5b` | Chat model alias |

Example:

```bash
RAG_TOP_K=2 RAG_SIMILARITY_THRESHOLD=0.4 python main.py
```

## Project structure

- `main.py` — CLI and model lifecycle management
- `rag_pipeline.py` — ingestion and query orchestration
- `document_loader.py` — recursive text/Markdown/PDF discovery, PDF text extraction, and word-safe chunking
- `embeddings.py` — Foundry embedding client wrapper
- `foundry_runtime.py` — lazy SDK/runtime manager
- `retriever.py` — cosine-similarity ranking
- `llm.py` — grounded chat generation, extractive validation, and response formatting
- `database.py` — SQLite persistence and atomic replacement
- `evaluate.py` — retrieval evaluation and smoke tests
- `evaluation_cases.example.json` — optional sample-only cases for the bundled demo documents
- `tests/test_core.py` — unit tests

## Knowledge base used for the demo

The included demo knowledge base contains five focused documents, matching the lower bound of the plan's recommended 5–10 document collection:

- `what_is_ai.txt` — artificial intelligence concepts and examples
- `machine_learning_basics.md` — learning from data, training, and prediction
- `generative_ai_basics.md` — generative AI and foundation models
- `classification_basics.txt` — binary and multiclass classification
- `ai_in_everyday_life.pdf` — text-based PDF covering everyday AI applications

## Demo plan

1. Run `python main.py --reindex` and show that five documents are embedded and stored.
2. Run `python evaluate.py --cases evaluation_cases.example.json` to demonstrate repeatable direct, paraphrased, and unsupported retrieval checks.
3. Start `python main.py` and ask `What is binary classification?` to show a grounded answer and source.
4. Ask `How does a computer learn from examples instead of explicit rules?` to demonstrate semantic paraphrase retrieval.
5. Ask `Who won the 2022 FIFA World Cup?` to demonstrate the insufficient-context guardrail.
6. Enter a blank question to demonstrate input validation, then `quit` for clean model shutdown.

## Lessons learned

- Document and query embeddings must come from the same model; storing the embedding model name in metadata prevents silent index incompatibility.
- Word-safe chunk boundaries and a modest overlap preserve context without cutting terms in half.
- Retrieval thresholds require both positive and negative evaluation cases; top-K ranking alone does not prove relevance.
- Source labels should come from retrieved database records rather than model-generated text.
- Streaming APIs can emit metadata-only events, so response parsing must tolerate empty choices and content.
- Small local models may ignore prompt-only grounding rules; extractive validation prevents unsupported expansions from reaching the user.
- Building the replacement index before the database transaction preserves the previous working index if embedding fails.

## Responsible output behavior

1. Keep the existing Top-K retrieval and similarity threshold.
2. Before generating an answer, ask the local model for exactly `ANSWERABLE` or `NOT_ANSWERABLE`. Only an exact `ANSWERABLE` permits generation; explanations, malformed responses, and uncertainty are rejected.
3. The local model selects numbered evidence passages. Code constructs the answer from those exact passages, preventing invented prose from becoming the answer. The old keyword-based substitution of a related sentence has been removed.
4. A separate model call verifies that the selected answer explicitly and completely addresses the question. Only exact `SUPPORTED` is accepted. Partial answers, missing essential facts, and a definition substituted for a specific requested fact are rejected.
5. Rejections return exactly `I don't have enough information in the provided documents to answer this question.` with no sources. Runtime failures remain explicit local-model error messages, also without sources.

All stages run locally with temperature 0. Strict parsing applies even at high retrieval scores. Source display for accepted answers retains the existing retrieved-file behavior.

Run the optional real-model regression tests on a host with Foundry Local device access:

```bash
python main.py --reindex
RAG_LIVE_TESTS=1 python -m unittest discover -s tests -v
```

Without this variable, real-model tests are skipped and the offline tests exercise parsing, call ordering, rejection, and source suppression using controlled model responses. Real-model tests check the actual classifier, passage selector, and verifier, including `Who invented binary classification?`.

## Limitations

- Answerability and final verification use the same small local model in separate calls. These checks reduce risk but cannot prove semantic correctness for every question. False negatives are intentionally preferred over unsupported answers.
- Positive answers require three local calls, so latency increases. The 0.5B model may reject answerable questions on longer retrieved context and may misjudge relevance; passing the small regression set is not a guarantee for arbitrary documents. The final answer consists only of selected verbatim passages; malformed selections and uncertain decisions are rejected.
- Prompts and evaluation cases are English-oriented; other languages are not validated.
- Sources identify retrieved files, not exact page references or sentence-level attribution.
- Initial setup/model downloads need internet. Verify cached-model operation without networking before an offline demonstration.
- Native GPU runtime failures (for example, `No supported adapters` in a sandbox) can terminate the process; run on a supported host with device access.

- The current retrieval is brute-force (suitable for small teaching datasets); it is not optimized for large corpora.
- PDF support is limited to extractable text; scanned/image-only PDFs are not OCR-processed and are skipped when they contain no readable text.
- The similarity threshold should be tuned using both answerable and unanswerable questions for best results.
- Small local models prioritize speed over fluency and may be less fluent than larger cloud models.

## References

- Build a RAG application with Foundry Local: https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app
- Foundry Local SDK reference: https://learn.microsoft.com/en-us/azure/foundry-local/reference/reference-sdk-current

## Submission hygiene

Exclude `.venv/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `__MACOSX/`, `._*`, local `.env` files, and generated `rag.db` (including journal/WAL files) from the submitted archive. `.gitignore` covers these files but does not remove them from a ZIP created from the entire folder. Keep the five sample documents, including the PDF, Python source, tests, requirements, README, and example evaluation JSON.

Default document/database paths are relative to the source directory, so the application can be launched from another working directory. Explicit relative environment-variable paths are relative to the working directory. Recreate the virtual environment on the receiving machine instead of copying it.
