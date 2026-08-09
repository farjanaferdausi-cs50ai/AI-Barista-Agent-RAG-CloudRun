# ☕ AI Barista Agent — RAG-Powered Assistant on Cloud Run

A Retrieval-Augmented Generation (RAG) AI agent that acts as a virtual barista for
a coffee shop. Built with Google's **Agent Development Kit (ADK)**, grounded in a
real menu dataset, wrapped in a **Streamlit** chat interface, and deployed to
**Google Cloud Run**.

Built while completing Google's official codelab, *"Deploy a RAG AI Agent in
Streamlit using Google ADK and Cloud Run,"* as part of the **Gen AI Academy APAC
(Hack2skill × Google Cloud), Cohort 3 — Track 1: Cloud Run + ADK + RAG**.

🌐 Live App Link: https://coffee-barista-827577333549.asia-southeast1.run.app/
---

## Build an AI Agent on Cloud Run with ADK and RAG : What it does

- Answers customer questions about drinks and pastries using natural language.
- Grounds every recommendation in an actual menu dataset — it will **not**
  invent items that don't exist on the menu.
- Understands allergen and dietary constraints (e.g. "I'm lactose intolerant")
  and filters recommendations accordingly.
- Asks a single clarifying question when a request is too vague, instead of
  guessing.
- Keeps a running, stateful conversation for the length of the browser session.

## Architecture

```
User (Streamlit chat) ──▶ InMemoryRunner ──▶ LlmAgent (Gemini 3.5 Flash)
                                                     │
                                                     ▼
                                          get_menu() tool ──▶ menu.json
```

Instead of pasting the entire menu into the model's system prompt, the agent
calls a small Python **tool** (`get_menu()`) on demand. The LLM only pays the
token cost for menu data when it actually needs it — this keeps prompts small
and answers grounded in real data rather than memorized guesses.

## Tech stack

| Layer            | Technology                                   |
|-------------------|-----------------------------------------------|
| LLM               | Gemini 3.5 Flash (via Vertex AI)              |
| Agent framework   | Google Agent Development Kit (ADK) — `LlmAgent`, `InMemoryRunner` |
| Retrieval         | Custom Python tool over a local JSON dataset  |
| UI                | Streamlit (stateful `st.session_state` chat)  |
| Deployment        | Google Cloud Run (source-based build, buildpacks) |
| IAM               | Dedicated least-privilege service account (`roles/aiplatform.user`) |

## Project structure

```
coffee-barista-agent/
├── menu.json          # RAG data source: menu items, prices, tags, allergens
├── agent.py           # ADK LlmAgent + get_menu() tool definition
├── app.py             # Streamlit chat UI + InMemoryRunner wiring
├── requirements.txt   # google-adk, streamlit
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
export GOOGLE_CLOUD_PROJECT=<project-852e862b-bbdd-4f9a-bbe>
export GOOGLE_CLOUD_LOCATION=global
streamlit run app.py
```

## Deploying to Cloud Run

```bash
# 1. Create a dedicated, least-privilege service account
gcloud iam service-accounts create barista-agent-sa \
  --display-name="Barista Agent Service Account"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:barista-agent-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# 2. Deploy straight from source — no Dockerfile required
gcloud run deploy coffee-barista \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --command "/cnb/lifecycle/launcher" \
  --args "sh,-c,python3 -m streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false" \
  --service-account "barista-agent-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=global
```

Cloud Run detects `requirements.txt`, builds a Python container automatically
via Buildpacks, and serves the Streamlit app — no Dockerfile or Procfile needed.

## Testing the RAG grounding

| Prompt                                     | Expected behavior                                             |
|---------------------------------------------|-----------------------------------------------------------------|
| "Recommend something strong and warm."      | Recommends an in-menu item (e.g. Espresso Solo).               |
| "Do you have a matcha frappuccino?"         | Politely declines — item is not on the menu.                   |
| "I'm lactose intolerant, what can I get?"   | Recommends only dairy-free items; excludes Croissant, Cappuccino, etc. |

## Future improvements

- **Firestore + Vector Search**: swap the static `menu.json` for a live
  Cloud Firestore collection, and retrieve items by semantic similarity
  (via `text-embedding-004` embeddings) instead of loading the full menu —
  enabling live menu updates without redeploying the container.
- **Persistent sessions**: replace `InMemoryRunner`'s in-memory state with a
  `SessionService` backed by Firestore or Redis so conversations survive
  page refreshes.

## Credits

Built by Farjana Ferdausi as part of the **Gen AI Academy APAC, Cohort 3**
(Hack2skill × Google Cloud), Track 1: *Cloud Run + ADK + RAG*, following
Google's official codelab:
[Deploy a RAG AI Agent in Streamlit using Google ADK and Cloud Run](https://codelabs.developers.google.com/codelabs/cloud-run/build-streamlit-rag-agent-google-adk-cloud-run).

👩‍💻 Author:

Farjana Ferdausi Aspiring AI & ML Engineering — Ostad (Batch-6),Bangladesh. Also studying AI Engineering & Data Science at CodeBasics,India. Artificial Intelligence Intern at CodeAlpha,India. Former HR Professional (14+ years) at Radisson Blu Dhaka Water Garden,Bangladesh.
