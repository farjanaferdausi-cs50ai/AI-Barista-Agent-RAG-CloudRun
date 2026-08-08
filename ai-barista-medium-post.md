# Building an AI Barista: A RAG-Powered Agent with Google ADK and Cloud Run

*How I built, grounded, and deployed an agent that recommends coffee without ever making up a menu item*

I just wrapped up a project that, on the surface, sounds almost silly: teaching an AI to recommend coffee. Underneath, though, it's a compact, complete example of the pattern behind almost every production AI agent shipping today — a language model, a tool it can call, a data source it's grounded in, and a deployment pipeline that puts it in front of real users.

"Agent" is one of the most overused words in AI right now, and I wanted to actually build one instead of just reading about the pattern. Not a chatbot that answers general-knowledge questions, but something narrower and more useful: an assistant that's given real tools, told explicitly what it's allowed to do with them, and shipped somewhere a real person could use it.

I built this as part of Track 1 of the **Gen AI Academy APAC** program (Hack2skill × Google Cloud) — a live one-hour workshop followed by a guided, hands-on codelab — working through Google's official tutorial on building a Retrieval-Augmented Generation (RAG) agent with the **Agent Development Kit (ADK)**, wrapping it in **Streamlit**, and deploying it to **Cloud Run**. The result is an "AI Barista" for a fictional coffee shop — an agent that answers questions like *"what's something strong and dairy-free?"* by pulling real answers from a real menu, and that flatly refuses to invent a drink that doesn't exist. The whole thing runs on Google Cloud for well under a dollar.

This post walks through how it's built, the design decisions baked into it, and what I took away from it as someone building toward a career in AI engineering.

## Why RAG, and Why Give the Agent a Tool?

Large language models are fluent, but they don't know your business. Ask a general-purpose model what's on a specific coffee shop's menu, and it will happily invent a plausible-sounding answer — a caramel frappuccino that doesn't exist, or a price that's simply guessed. That's the core problem Retrieval-Augmented Generation is built to solve: instead of trusting the model's memory, you ground its answers in a real, external source of truth at the moment it responds.

For a coffee shop, a made-up drink is a minor annoyance. For an agent recommending medication interactions, financial products, or anything with real consequences, that same confident invention is a liability instead of a quirk. The barista is a low-stakes way to practice a discipline that matters a lot more once the stakes go up.

There are a couple of ways to do that. The naive approach is to paste your entire dataset into the system prompt, every single time. For eight menu items, that's cheap. But real menus change, grow, and get more complex — and stuffing a large dataset into every prompt inflates token costs and latency on every single query, whether the user needs that data or not.

The more scalable pattern — and the one this project uses — is to give the agent a **tool**: a plain Python function it can decide to call when it actually needs the data. The model only pays for the menu's tokens when a menu question actually comes up. This is the same tool-calling pattern behind most serious production agents, whether they're checking a database, calling an API, or searching documents.

## Architecture at a Glance

The whole system breaks down into four pieces:

```
Streamlit chat UI  →  InMemoryRunner  →  LlmAgent (Gemini 3.5 Flash)
                                                │
                                                ▼
                                    get_menu() tool  →  menu.json
```

A user types a question into a Streamlit chat box. The ADK's `InMemoryRunner` routes that message to an `LlmAgent` powered by Gemini 3.5 Flash. When the model decides it needs menu data to answer, it calls the `get_menu()` tool, which reads a local `menu.json` file and hands the results back. The model then answers using only what that tool returned.

Nothing in that chain is hardcoded to always fetch the menu on every turn — the model itself decides, message by message, whether a question actually needs grounding data. That decision-making is what makes this an agent rather than a fixed script with an LLM bolted on.

## Step 1: A RAG Data Source That Doesn't Pretend to Be More Than It Is

The grounding source here is intentionally simple: a `menu.json` file with eight items, each with a name, description, price, a list of descriptive tags (`strong`, `cold`, `dairy-free`, `vegan`), and an allergens list.

```json
{
  "name": "Oat Milk Honey Latte",
  "description": "Creamy steamed oat milk with espresso and a touch of honey.",
  "price": 5.00,
  "tags": ["sweet", "hot", "dairy-free"],
  "allergens": []
}
```

The schema is deliberately narrow: five fields, no nesting beyond a couple of arrays. Tags and allergens are kept as two separate lists on purpose — a drink's *style* (strong, cold, sweet) and its *safety information* (dairy, wheat) answer two different kinds of questions, and collapsing them into one field would have made the agent's allergen-filtering instruction much harder to get reliably right.

A flat local file is obviously not how you'd run this in production — a real coffee shop needs to update prices and seasonal items without redeploying an app every time. But for a prototype, it removes an entire layer of setup and lets you focus on the agent logic first. It's a deliberate simplification, not a naive one, and it's one I later swap out (more on that below).

## Step 2: Building the Agent with Google's ADK

The Agent Development Kit is Google's open-source framework for building these tool-using agents, and the entire agent definition here comes down to two things: a tool function, and an `LlmAgent`.

The tool itself is almost aggressively simple:

```python
def get_menu() -> str:
    """Retrieves the coffee shop menu from menu.json."""
    try:
        with open("menu.json", "r") as f:
            menu_data = json.load(f)
            return json.dumps(menu_data)
    except Exception as e:
        return json.dumps({"error": f"Could not retrieve menu: {str(e)}"})
```

No embeddings, no vector math, just a function that reads a file and returns JSON as a string. What makes this count as *RAG* isn't the sophistication of the retrieval mechanism; it's that the model is required to go through it rather than answering from memory.

ADK itself supports more than this single-agent pattern — it's built around composable agent types, including workflow agents that chain multiple steps together and multi-agent setups where specialized agents hand off to each other. This project only needed one `LlmAgent` with one tool, but knowing that it's the same building block that scales up to more complex systems is what made the framework feel worth learning properly, rather than treating it as a one-off tutorial API.

The real engineering happens in the agent's instructions. I gave the `LlmAgent` five explicit rules: recommend only from what `get_menu()` returns, never suggest an item that isn't in that data, ask exactly one clarifying question when a request is vague instead of guessing, stay warm but professional, and cross-reference tags and allergens honestly, so a dairy-free request only ever surfaces items that are actually dairy-free.

That level of explicitness surprised me a little. It's tempting to assume that connecting a tool is enough, and the model will "figure out" how to use it responsibly. In practice, the instructions are doing as much safety work as the retrieval itself. They're what turns "a model with access to data" into "a model that reliably tells the truth about that data." Access and instruction turn out to be two separate problems, and RAG on its own only solves the first one.

```python
barista_agent = LlmAgent(
    name="barista_agent",
    model="gemini-3.5-flash",
    instruction="""You are a friendly barista...
    1. Recommend items ONLY from the menu returned by get_menu().
    2. Do NOT recommend anything not present in the menu.
    3. Ask exactly ONE clarifying question if a request is vague.
    4. Stay warm, but professional.
    5. Ground recommendations honestly in tags and allergens.
    """,
    tools=[get_menu],
)
```

The agent is then wrapped in an `App` object, which is what the ADK runner actually executes.

## Step 3: A Stateful Chat, Wrapped in Streamlit

Streamlit turns this from a script into something a customer could actually use. It's a natural fit for this kind of project because it's pure Python — there's no separate frontend codebase and no JavaScript build step, just Python functions that render UI as the script re-runs top to bottom on every interaction.

The interesting plumbing here is state management: `st.session_state` holds a unique session ID, an `InMemoryRunner` instance, and the running list of chat messages, all scoped to a single browser tab.

Every new user message goes through the same short pipeline: append it to the visible chat history, hand it to the runner asynchronously (Streamlit's execution model is synchronous, so this gets wrapped in `asyncio.run()`), collect the text parts out of the returned events, and render the response.

The sidebar renders the live menu, name, price, tags, and allergen warnings, pulled from the same `menu.json` the agent uses, so a customer can sanity-check what the agent tells them against the actual list.

One limitation is worth calling out honestly: `st.session_state` is entirely in-memory. Close the tab or refresh the page, and the conversation is gone. That's a fine trade-off for a demo, but it's the first thing I'd change for a real deployment, and it's directly connected to the bigger upgrade path below.

## Step 4: Shipping It to Cloud Run, From Source, No Dockerfile

Deployment comes down to a single command:

```bash
gcloud run deploy coffee-barista \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --service-account "barista-agent-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=$PROJECT_ID
```

No `Dockerfile`, no `Procfile`. Cloud Run's buildpacks detect `requirements.txt` and the Python source, and build a runnable container automatically, which is a genuinely nice on-ramp when you want something deployed without first becoming a container-tooling expert. Buildpacks work by inspecting the source directory, recognizing the language and framework from files it finds, and assembling a production-ready image without anyone writing container instructions by hand — the same category of tooling that platforms like Heroku popularized, brought natively into Cloud Run.

The part I paid the most attention to, though, was identity, not code. Rather than deploying under the default Compute Engine service account, which typically carries broad Editor-level permissions, the setup creates a dedicated `barista-agent-sa` service account and grants it exactly one role: `roles/aiplatform.user`, enough to call Gemini and nothing more. If this app ever had a security bug, the blast radius is deliberately small. It's a simple habit, but one worth carrying into every agent I deploy from here on: give it the access it needs, not the access that happens to be convenient.

## Step 5: Testing the Guardrails, Not Just the Happy Path

Once deployed, testing wasn't just "does it respond," it was specifically about whether the grounding held up:

- **A normal request** ("something strong and warm") correctly surfaces an in-menu item like the Espresso Solo.
- **An out-of-menu trap** ("do you have a matcha frappuccino?") gets a polite decline, not a hallucinated yes.
- **An allergen-aware request** ("I'm lactose intolerant, what can I get?") filters down to genuinely dairy-free items only, excluding anything carrying a dairy allergen tag.

That second test case matters more than it might seem. A demo that only ever answers questions it can answer correctly doesn't tell you much. The valuable test is deliberately trying to trick the agent into making something up, and confirming it holds the line anyway. Only after all three held up did the deployment actually feel finished — a working demo and a *trustworthy* demo turned out to be two different bars to clear.

## Where I'd Take This Next: Firestore and Vector Search

The codelab's optional extension points at the obvious production upgrade: move `menu.json` into **Cloud Firestore**, generate a text embedding for each item using Vertex AI's `text-embedding-004` model, and store that embedding as a vector field alongside the item. (An embedding, for anyone new to the term, is just a list of numbers representing the *meaning* of a piece of text — items with similar meanings end up with numerically similar vectors, which is what makes searching by meaning, instead of exact keywords, possible.)

At query time, instead of loading the entire menu, the `get_menu()` tool becomes a semantic search: embed the user's query, and run a nearest-neighbor vector query against Firestore using cosine distance to pull back only the handful of most relevant items. A shop manager could add a new seasonal drink directly in Firestore, and the agent would be able to recommend it immediately, with no redeploy required.

That's the natural next iteration I want to build on top of this, not because the JSON version is "wrong," but because it makes the trade-off visible. A local file is the right choice while you're proving out agent behavior; a live, embedded, queryable database is the right choice once the data needs to change without a deploy cycle.

## What This Project Actually Taught Me

The flashy part of "AI agents" is the model. The part that actually determines whether an agent is trustworthy is almost entirely everything *around* the model: what data it's allowed to see, how explicitly it's told to stay inside those bounds, what permissions its deployment carries, and whether anyone bothered to test the cases where it's tempted to make something up.

Building this end-to-end, data source, agent, interface, deployment, and a deliberate attempt to break it, gave me a much more complete picture of "agentic AI" than any single tutorial on prompting alone could have. It's a small project, but the pattern scales: LLM, tool, grounding data, and a deployment discipline that doesn't hand out more access than the job needs. That's the shape of most agents I expect to be building a lot more of going forward.

It also fits a pattern I keep noticing across everything I'm working on right now, whether it's a computer vision pipeline, a fine-tuning experiment, or this: the model itself is rarely the hard part anymore. The hard part is the engineering built around it — what data it's allowed to see, the constraints it operates under, and the discipline of testing for the ways it can fail, not just the ways it can succeed. That's the muscle I'm actually trying to build, one small project at a time.

## Try It Yourself

The full code, `menu.json`, `agent.py`, `app.py`, and the deployment steps, is on my GitHub. If you want to build your own version from scratch, Google's original codelab is a genuinely well-paced way to do it, and it's where this project started.
