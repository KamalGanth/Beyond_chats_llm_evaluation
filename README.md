# Beyond Chats – LLM Response Evaluation Pipeline

## Overview

This project implements a **end-to-end LLM response evaluation pipeline**. The goal is to systematically evaluate the quality of Large Language Model (LLM) responses using **multiple complementary evaluation techniques**, ranging from lightweight statistical methods to advanced LLM-based judges.

The pipeline is designed to be:

* **Modular** – each stage is independently extensible
* **Robust** – handles malformed real-world JSON inputs
* **Scalable** – supports fast heuristics as well as LLM-based evaluation
* **Reproducible** – deterministic preprocessing + clear metrics

It is particularly suitable for:

* RAG (Retrieval-Augmented Generation) evaluation
* LLM benchmarking




##  LLM Evaluation Pipeline – Architecture

The following diagram illustrates the end-to-end architecture of the LLM Evaluation Pipeline, 
from raw input ingestion to final reporting.


![llm Architecture diagram](<assets/llm_architecture.png>)

## Detailed Pipeline Explanation

### 1. Input Layer

The pipeline consumes **two independent inputs per evaluation**:

* **Conversation JSON**

  * Contains a full chat history between user and AI
  * May be malformed (logs, trailing commas, bad quotes)

* **Context Vector JSON**

  * Retrieved documents or chunks used by the LLM
  * Typically produced by a vector database in RAG systems

Each `(conversation, context)` pair represents **one evaluation sample**.



### 2. Preprocessing Layer (`preprocess.py`)

Purpose:

* Convert real-world, noisy inputs into a **clean evaluation unit**

Key responsibilities:

* Multi-pass JSON cleaning (strict → regex repair → AST fallback → emergency extraction)
* Extract **last user query** and **last AI response**
* Select top-K context chunks

Output:

```json
{
  "user_message": "...",
  "ai_response": "...",
  "context": ["chunk1", "chunk2", ...]
}
```

Why only the last turn?

> Relevance and hallucination are properties of a **specific response** with respect to retrieved context. Earlier turns add noise and are excluded by design.



### 3. Evaluation Layer (`evaluate.py`)

This is the **core of the system**. The same processed input can be evaluated using **multiple strategies**, selectable at runtime.

Each method produces:

* Relevance score
* Completeness score
* Hallucination score
* Latency (ms)
* Cost estimate



##  Evaluation Techniques – Why These Choices?

###  1. TF-IDF (Statistical Baseline)

**What it does:**

* Measures lexical overlap between response and context

**Why included:**

* Extremely fast
* Zero API cost
* Strong baseline for factual overlap

**Limitations:**

* No semantic understanding
* Fails on paraphrasing

**Best for:**

* Quick sanity checks
* Large-scale batch evaluation


###  2. Embedding Similarity (Semantic Baseline)

**What it does:**

* Uses sentence embeddings to compute semantic similarity

**Why included:**

* Captures meaning beyond word overlap
* Much stronger than TF-IDF

**Limitations:**

* Still correlation-based
* Cannot judge reasoning quality

**Best for:**

* Semantic grounding checks
* Medium-scale evaluation


###  3. LLM Judge (Gemini)

**What it does:**

* Uses an LLM to judge answer quality

**Why included:**

* Closest to human evaluation
* Can assess nuance and reasoning

**Limitations:**

* Higher latency
* API cost

**Best for:**

* High-stakes evaluation
* Spot-checking critical outputs



###  4. DeepEval (Standardized LLM Evaluation)

**What it does:**

* Uses industry-standard metrics:

  * Answer Relevancy
  * Faithfulness (hallucination)

**Why included:**

* Research-grade framework
* Structured, explainable judgments
* Widely used in RAG evaluation research

**Advantages over raw LLM judge:**

* Consistent scoring
* Built-in reasoning explanations
* Metric-level thresholds

**Best for:**

* Benchmarking
* Research & academic work


##  Latency & Cost Tracking

Each evaluation method is wrapped with a timing utility:

* Measures execution time in milliseconds
* Enables scalability analysis

Cost estimation:

* Token-based heuristic
* Highlights trade-offs between cheap heuristics and LLM-based judges



## Project Structure
```
Beyond_chats_llm_evaluation/
│
├── main.py              # Unified entry point
├── preprocess.py        # Robust preprocessing
├── evaluate.py          # Evaluation orchestration
├── report.py            # Markdown report generation
│
├── utils/
│   ├── cleaner.py       # Fault-tolerant JSON ingestion
│   ├── scorer.py        # TF-IDF / Embedding scoring
│   ├── timer.py         # Latency measurement
│   ├── cost.py          # Cost estimation
│   └── deepeval_scorer.py
│
├── data/
│   ├── sample-chat-conversation-01.json
│   ├── sample_context_vectors-01.json
|
│___output/
├── requirements.txt
└── README.md
```



###  Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/KamalGanth/Beyond_chats_llm_evaluation.git
cd Beyond_chats_llm_evaluation
```



### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```



### 3. Install Dependencies

```bash
pip install -r requirements.txt
```



### 4. Set API Keys (If Needed)

Create .env file and include 

GOOGLE_API_KEY=Add your gemini api key (https://ai.google.dev/gemini-api/docs/api-key)

##  How to Run the Pipeline

```bash
python main.py
```

You will be prompted to:

1. Select a conversation file
2. Select a context file
3. Choose evaluation method

The pipeline will automatically:

* Preprocess input
* Run evaluation
* Generate report



## Outputs

Generated at runtime (not committed to Git):

* `data/processed_input.json`
* `data/raw_evaluation.json`
* `data/final_report.md`



## Design Philosophy

This system intentionally combines:

* **Cheap heuristics** (TF-IDF)
* **Semantic methods** (Embeddings)
* **Human-like judgment** (LLM Judge)
* **Standardized evaluation** (DeepEval)

This mirrors **real-world LLM evaluation stacks** used in production and research.


## Future Extensions

* Batch evaluation across datasets
* Side-by-side method comparison
* Dashboard visualization
* Confidence grading (PASS / WARN / FAIL)



## Author

**Kamal Ganth**
LLM Evaluation pipeline 


