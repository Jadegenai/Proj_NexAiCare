# NexAiCare — AI-Powered Healthcare Platform

> **Six Integrated AI Modules in One Unified Application**
> Developed by **Jade Global**

---

## Overview

NexAiCare is a production-ready Streamlit application that unifies six AI-powered healthcare modules into a single platform. It leverages LangChain, OpenAI, ChromaDB, and SQLite to deliver clinical decision support, claims auditing, and consultation summarization — all behind an intuitive, dashboard-driven UI.

**Phase 1 (Active):** Modules 1–3 are fully implemented.
**Phase 2 (Planned):** Modules 4–6 are scaffolded with "Coming Soon" pages.

---

## Modules

| #  | Module                    | Status       | Description |
|----|---------------------------|--------------|-------------|
| 1  | Medical Assistant AI      | ✅ Active    | RAG-powered knowledge base backed by the Merck Medical Manual (4 000+ pages). Ask clinical questions in natural language and receive grounded, source-cited answers. |
| 2  | Claim Audit AI            | ✅ Active    | Natural-language SQL agent for healthcare claims auditing. Includes input/output guardrails, context memory for follow-up queries, and PII/PHI protection. |
| 3  | Consultation Notes AI     | ✅ Active    | AI-powered summarization of doctor-patient conversations into structured clinical notes (Patient Concern → Findings → Action Plan). |
| 4  | PII/PHI Monitor           | 🔜 Phase 2  | Continuous HIPAA/GDPR/CCPA compliance monitoring with automated PII/PHI detection and redaction. |
| 5  | Medical Coding AI         | 🔜 Phase 2  | Automated CPT/ICD-10 coding suggestions with audit trails from EHR data. |
| 6  | Clinical Diagnostic AI    | 🔜 Phase 2  | Medical imaging captioning (X-ray, CT, MRI) with differential diagnosis suggestions. |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NexAiCare Platform                           │
│                     (Streamlit Application)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │   SIDEBAR     │     ┌──────────────────────────────────────────┐ │
│  │              │     │           MAIN CONTENT AREA               │ │
│  │ Jade Global  │     │                                          │ │
│  │ Logo         │     │  ┌────────────────────────────────────┐  │ │
│  │              │     │  │        📊 DASHBOARD                 │  │ │
│  │ ───────────  │     │  │  KPI Cards │ Charts │ AI Insights  │  │ │
│  │              │     │  │  Claims Table with Filters          │  │ │
│  │ 📊 Dashboard │     │  └────────────────────────────────────┘  │ │
│  │ 🩺 Module 1  │     │                                          │ │
│  │ 🔍 Module 2  │     │  ┌────────────────────────────────────┐  │ │
│  │ 📝 Module 3  │     │  │   🩺 MEDICAL ASSISTANT AI (RAG)    │  │ │
│  │ 🛡️ Module 4  │     │  │                                    │  │ │
│  │ 💻 Module 5  │     │  │  PDF ──► Chunks ──► Embeddings     │  │ │
│  │ 🔬 Module 6  │     │  │         ──► ChromaDB Vector Store  │  │ │
│  │ ⚙️ Settings  │     │  │  User Query ──► Retriever (top-k)  │  │ │
│  │              │     │  │         ──► LLM + Context ──► Answer│  │ │
│  │ ───────────  │     │  └────────────────────────────────────┘  │ │
│  │ 🟢 API OK    │     │                                          │ │
│  │              │     │  ┌────────────────────────────────────┐  │ │
│  │ ───────────  │     │  │   🔍 CLAIM AUDIT AI (SQL Agent)    │  │ │
│  │ © Jade Global│     │  │                                    │  │ │
│  └──────────────┘     │  │  Query ──► Input Guardrail (0-3)   │  │ │
│                       │  │       ──► SQL Agent (LangChain)    │  │ │
│                       │  │       ──► Output Guardrail         │  │ │
│                       │  │       ──► Context Memory           │  │ │
│                       │  └────────────────────────────────────┘  │ │
│                       │                                          │ │
│                       │  ┌────────────────────────────────────┐  │ │
│                       │  │   📝 CONSULTATION NOTES AI          │  │ │
│                       │  │                                    │  │ │
│                       │  │  Conversation ──► System Prompt    │  │ │
│                       │  │       ──► LLM (GPT-4o-mini)       │  │ │
│                       │  │       ──► Structured Summary       │  │ │
│                       │  └────────────────────────────────────┘  │ │
│                       └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

                        TECHNOLOGY STACK

    ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
    │ LangChain │  │  OpenAI   │  │ ChromaDB  │  │  SQLite   │
    │ LangGraph │  │ GPT-4o-   │  │  Vector   │  │  Claims   │
    │  Agents   │  │   mini    │  │   Store   │  │    DB     │
    └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
          │              │              │              │
    ┌─────┴──────────────┴──────────────┴──────────────┴─────┐
    │              Streamlit / Python Runtime                  │
    └─────────────────────────┬───────────────────────────────┘
                              │
    ┌─────────────────────────┴───────────────────────────────┐
    │         Deployment: Streamlit Cloud / Snowflake          │
    └─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer          | Technology                                    |
|----------------|-----------------------------------------------|
| Frontend       | Streamlit, Plotly, Custom CSS                 |
| AI / LLM       | OpenAI GPT-4o-mini, LangChain                |
| Embeddings     | OpenAI `text-embedding-ada-002`               |
| Vector Store   | ChromaDB (persistent on disk)                 |
| SQL Agent      | LangChain `create_sql_agent` + SQLite         |
| Database       | SQLite (`hc_data.db` — 100 claims records)    |
| PDF Processing | PyMuPDF (PyMuPDFLoader)                       |
| Tokenizer      | tiktoken (`cl100k_base`)                      |

---

## Project Structure

```
NexAiCare/
├── app.py                              # Main Streamlit application (single file)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── hc_data.db                          # SQLite claims database (100 records)
├── medical_diagnosis_manual.pdf        # Merck Medical Manual (4 000+ pages)
├── finetuning_medical_testing.csv      # 20 doctor-patient conversation samples
├── .streamlit/
│   └── config.toml                     # Streamlit theme configuration
├── Healthcare_db/                      # ChromaDB vector store (created at runtime)
├── Jade_Global_Logo_Dark_Mode.PNG      # Logo (optional — text fallback provided)
├── Jade_Global_Logo_Light_Mode.PNG     # Logo (optional — text fallback provided)
├── Healthcare_AI_Specialist.ipynb      # Module 1 reference notebook
├── Healthcare_Audit_Chatbot_Solution_Notebook.ipynb  # Module 2 reference notebook
└── Doctor_Interaction_Summary.ipynb    # Module 3 reference notebook
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <repo-url>
cd NexAiCare
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

**Option A — Environment Variable:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
# Optional: export OPENAI_API_BASE="https://custom-endpoint"
```

**Option B — Streamlit Secrets (for Cloud deployment):**
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-key-here"
# OPENAI_API_BASE = "https://custom-endpoint"
```

**Option C — In-App Settings:**
Navigate to ⚙️ Settings and enter your key directly.

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Deployment

### Streamlit Cloud

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Select the repo, branch, and `app.py` as the main file.
4. Add `OPENAI_API_KEY` in the **Secrets** section.
5. Deploy.

### Snowflake (Streamlit in Snowflake)

1. Upload data files (`hc_data.db`, `medical_diagnosis_manual.pdf`, `finetuning_medical_testing.csv`) to a Snowflake stage.
2. Adapt file paths in `app.py` to reference staged files.
3. Replace `SQLDatabase.from_uri("sqlite:///...")` with Snowflake connector if using Snowflake tables.
4. Deploy via Snowflake's Streamlit app interface.

---

## Process Flow

### Module 1 — Medical Assistant AI (RAG Pipeline)

```
User Question
      │
      ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  PDF Loader  │────►│  Text Split  │────►│  Embeddings  │
│ (PyMuPDF)    │     │ (512 tokens) │     │  (OpenAI)    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                         ┌──────────────┐
                                         │  ChromaDB    │
                                         │ Vector Store │
                                         └──────┬───────┘
                                                 │
User Query ──► Embedding ──► Similarity Search ──┘
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │  Top-k Docs  │
                                         └──────┬───────┘
                                                 │
                                    Context + Question
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │  GPT-4o-mini │
                                         │  + System    │
                                         │    Prompt    │
                                         └──────┬───────┘
                                                 │
                                                 ▼
                                          Grounded Answer
                                          + Source Pages
```

### Module 2 — Claim Audit AI (SQL Agent + Guardrails)

```
User Query
      │
      ▼
┌──────────────────┐
│  INPUT GUARDRAIL │
│  (Intent 0-3)    │
└──────┬───────────┘
       │
  ┌────┼────┬────┐
  0    1    2    3
  │    │    │    │
  ▼    ▼    ▼    ▼
Esc  Exit  OK  Block
              │
              ▼
┌──────────────────┐     ┌──────────────┐
│  Context Memory  │────►│  SQL Agent   │
│  (chat history)  │     │ (LangChain)  │
└──────────────────┘     └──────┬───────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │ OUTPUT GUARDRAIL │
                      │  (SAFE / BLOCK)  │
                      └──────┬───────────┘
                             │
                        SAFE │ BLOCK
                             │
                             ▼
                      Final Response
```

### Module 3 — Consultation Notes AI (Summarization)

```
Doctor-Patient Conversation
           │
           ▼
┌───────────────────────┐
│    System Prompt       │
│  (Clinical format:    │
│   Concern / Findings  │
│   / Action Plan)      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│     GPT-4o-mini       │
│   (temp=0.2, 400 tok) │
└───────────┬───────────┘
            │
            ▼
   Structured Clinical
        Summary
```

---

## Data Sources

| File | Description | Size |
|------|-------------|------|
| `hc_data.db` | SQLite database with 100 healthcare claims records across 8 departments, 20+ providers, and 3 claim statuses. | 32 KB |
| `medical_diagnosis_manual.pdf` | Merck Medical Manual — comprehensive medical reference with 4 000+ pages covering disorders, diagnoses, and treatments. | 20 MB |
| `finetuning_medical_testing.csv` | 20 doctor-patient conversation/summary pairs for testing consultation summarization quality. | 5 KB |

---

## License

Proprietary — Jade Global. All rights reserved.
