# AI support agent

A customer support agent for Fernhill, an invented B2B SaaS company. It answers from a help center, checks who it is talking to before it touches an account, and can refund an invoice, change a contact field or hand the case to a person. It streams every step to the client as a typed event.

The project follows a brief that asks for streaming chat, live voice, multi-step tool chains, confidence guardrails and real-time observability. It is being built in four phases, and this README changes with every push.

## Status

Phase 2 of 4, deployed. Try it at **https://fernhill-support.vercel.app**. The chat UI and the backend run on hosted models. There is no voice mode yet.

| What the brief asks for | State today |
|---|---|
| Streaming chat UI with live step status | Done, in Next.js over a WebSocket. Each tool shows as a step while it runs and with its result once it is done. |
| Live voice with waveform, transcripts and interruption | Not started. Phase 3. |
| Observability drawer | Done. Per turn: end-to-end latency, first token, tokens, estimated cost, a timeline of every model and tool call, retrieval scores, the grounding verdict, and the full prompts with PINs masked. |
| Vector database with custom chunking | Done. pgvector and full-text search fused in SQL, a chunker that follows the heading tree, a reranker. |
| Confidence guardrails that route to a person | Done. A gate on the reranker's score, and a second model that checks each answer. |
| A chain of three or more tool calls | Done. Six tools, with the business rules in code. |
| Tool start and finish streamed to the UI | Done. |
| Fallback states for timeouts, rate limits and tool failures | Done. The UI shows a retry, a switch to the backup model, a failed turn with Try again, a stopped turn with Resume, and a dropped connection. A panel can inject each failure on demand. |
| One memory across text and voice | A session is stored under its ID. The browser keeps the ID, so a reload or a second tab continues the conversation. Voice does not exist yet. |
| Deployed | Yes. The frontend is on Vercel, the backend and Postgres with pgvector on Railway. |

## How a turn runs

```mermaid
flowchart LR
    C["Browser or terminal"] <-->|"WebSocket, numbered events"| H["Session hub"]
    H --> B["begin"]
    subgraph L["Agent loop, LangGraph"]
        B --> A["agent"]
        A -->|"tool call"| T["tools"]
        T --> A
        A -->|"final answer"| K["check"]
        A -->|"step limit"| E["escalate"]
        T -->|"hand off"| E
        K -->|"unsupported"| E
    end
    T <--> P[("Postgres: mock CRM, pgvector, checkpoints, event log")]
```

1. `begin` stores the customer's message. If the last turn died half way, it first repairs the conversation, so the model never sees a tool call without a result.
2. `agent` calls the model with the six tools. The model makes one tool call per step.
3. `tools` runs the call inside a guard. The guard emits `tool.started` and `tool.finished`, enforces a timeout, retries a failure that may pass, and turns any other failure into an error the model can read.
4. `check` sends the final answer and its evidence to a second model. An answer with an unsupported statement is withdrawn.
5. `escalate` opens a ticket and replies with fixed wording. No model writes that text.

LangGraph saves the conversation in Postgres after every step. The event log is in Postgres too, numbered per session. A turn belongs to the session and not to the socket, so it finishes and is stored even when the client has gone.

Chat and voice are meant to be two clients of this one pipeline. The turn runner already takes a mode, text or voice, and the system prompt changes with it.

## The six tools

| Tool | What it does | Rules the code enforces |
|---|---|---|
| `verify_identity` | Checks an account number and PIN | Three wrong PINs lock the conversation and open a ticket. The count never resets. |
| `get_account` | Reads plan, contract status, seats, renewal date and recent invoices | Needs a verified customer |
| `search_knowledge_base` | Searches the help center, limited to passages the customer's plan may see | Returns a band that decides what happens to the answer. See the scope gate below. |
| `issue_refund` | Refunds one invoice | 14 days for Pro monthly, 30 days otherwise, 90 days for a duplicate charge. Up to $500 on Pro and $2,000 on Enterprise. Anything above goes to the billing team as a ticket. The invoice has to be paid and not yet refunded. No refunds on Free, on a suspended account or on a past-due one. |
| `update_account_field` | Changes the billing email, contact name, contact phone or marketing opt-in | Validates the value. No other field can change. |
| `create_escalation_ticket` | Hands the case to a person | One ticket per call, even when the call is retried |

The model cannot talk its way around these. The limits are constants in `crm/policy.py`, the refund runs in one transaction that locks the customer row, and the account tools refuse until `verify_identity` has succeeded in this conversation.

## Guardrails

### Scope gate

Every search ends in one of five bands. The band comes from the reranker's best score among the passages the customer may see, and from its best score among passages written for other plans.

| Band | When | What happens |
|---|---|---|
| `answer` | The best visible passage scores at or above `GATE_HIGH` | The answer streams as it is written and is checked afterwards |
| `check` | The score is between `GATE_LOW` and `GATE_HIGH` | The answer is held back until the grounding check passes |
| `verify` | The visitor is not verified, and a passage for another plan answers much better than anything public | The agent asks for the account number and PIN. It gets nothing from the hidden passage. |
| `restricted` | The customer is verified, and a passage for another plan is a confident match | The agent says the topic belongs to another plan |
| `escalate` | Nothing scores above `GATE_LOW` | A ticket, and the hand-off text |

The thresholds belong to the reranker. Change the reranker and all three settings need a new calibration with `supportagent.evals.retrieval`.

### Grounding check

A relevance score cannot tell a question the documents answer from a question they only come close to. "What was your uptime last month?" finds the page about the uptime commitment and scores high. So every final answer also goes to a judge model, together with the tool results of the turn, the customer's account and a list of what the tools can do.

The judge returns JSON. An answer with any unsupported statement is withdrawn and the case goes to a person. If the judge is down, the answer is withheld and the turn fails as retryable. The retry reruns the check and nothing else.

The request wraps the evidence, the customer's words and the reply in tags that carry a one-time code, and it marks every line of customer and reply text with who wrote it. A customer cannot type something that reads as evidence.

### PINs

The model has to see the PIN to pass it to `verify_identity`. Nothing else should. A redactor masks it in every event, trace and ticket, and removes it from the stored conversation once the tool has used it. While a visitor is unverified, every number of 4 to 8 digits they type is masked in what the backend stores and sends.

One gap remains. A PIN typed in a message stays in the saved conversation until the model calls the tool.

## When things fail

- A model call has four time limits: until the first token, between two chunks of the stream, for one attempt, and for the whole call with its retries. Rate limits and server errors are retried with backoff, and a `retry-after` header is obeyed. A billing error is not retried, because waiting does not clear it.
- Each model has its own retry budget. When the first model is exhausted, a fallback model on another provider takes over, if one is configured.
- An error that arrives in the middle of a stream is handled like one at the start. The half-written text is dropped and the client gets an `llm.reset` event.
- A tool that keeps failing produces an error the model reads. The model then says so or opens a ticket.
- A client that reconnects sends the last event number it saw. It gets the stored events it missed, then the live stream, each exactly once.
- `turn.cancel` stops a running turn. `turn.retry` resumes a failed turn from its last saved step.
- With `FAULT_INJECTION=true` a client can queue simulated timeouts, rate limits, outages and tool failures for its own session. That is how the failure paths get shown in a demo.
- One address can start 30 conversations an hour and a conversation takes 60 messages, both settings. Coming back to an existing conversation is not counted. The page says so when the server turns it away.

## Run it

You need [uv](https://docs.astral.sh/uv/), Docker, and an OpenAI-compatible chat endpoint. uv installs Python 3.12 by itself. So far this has only run on Linux.

```sh
git clone https://github.com/raghav-rathi/AISupportAgent.git
cd AISupportAgent
cp .env.example .env        # set POSTGRES_PASSWORD, and the same password inside DATABASE_URL
scripts/dev-db.sh up        # Postgres 17 with pgvector, on 127.0.0.1:54329
cd backend
uv sync
uv run python -m supportagent.cli.ingest      # applies the migrations, loads the help center
uv run python -m supportagent.cli.preflight   # says what is missing
uv run pytest -q
```

The tests need the database and no model. They run the real graph, tools and retrieval with a scripted model in place of an LLM. The first run downloads two small ONNX models for embedding and reranking.

Any Postgres with the `vector` extension works in place of the Docker script. Point `DATABASE_URL` at it.

### Pick a model

The agent and the judge each need a chat endpoint. There are two ways to give them one.

Through OpenRouter. Set `OPENROUTER_API_KEY`. The example file is filled in for this. Start with the check below. It reads each provider's recent uptime from OpenRouter's public listing, then sends one small request to every model and prints what came back.

```sh
uv run python -m supportagent.cli.gateway_check ../evals/candidates.hosted.json
```

Through a local server such as Ollama. Replace the `LLM__` and `JUDGE__` lines of `.env` with these, and leave `RERANK_MODEL` and the three `GATE_` lines out:

```sh
LLM__MODEL=qwen3.8:27b-q4_K_M
LLM__BASE_URL=http://127.0.0.1:11434/v1
LLM__API_KEY=ollama
LLM__EXTRA_BODY={"reasoning_effort":"none"}
JUDGE__MODEL=qwen3.8:27b-q4_K_M
JUDGE__BASE_URL=http://127.0.0.1:11434/v1
JUDGE__API_KEY=ollama
JUDGE__EXTRA_BODY={"reasoning_effort":"none"}
LLM_FIRST_TOKEN_TIMEOUT_S=120
LLM_TOTAL_TIMEOUT_S=240
LLM_DEADLINE_S=300
```

The three limits above suit a 24 GB GPU. On a slower machine raise `LLM_STALL_TIMEOUT_S` too. It is the longest silence allowed inside a stream, 10 seconds by default, and a local server sends a tool call in one piece, so the stream is silent for as long as the call takes to write.

A judge on the same model as the agent is good enough to try things out. In production it would approve its own mistakes, so the example file puts the judge on another provider.

### Talk to it

In a terminal:

```sh
uv run python -m supportagent.cli.chat --verbose      # the agent inside this process

../scripts/dev-server.sh up                           # or a server on 127.0.0.1:8787
uv run python -m supportagent.cli.ws_chat --verbose   # and a client on its WebSocket
../scripts/dev-server.sh down
```

Both clients print every event: tool steps, the band of each search, the verdict of the grounding check, retries, tickets.

In the browser. You need Node 20 or newer, and the backend running as above. Add the page's address to the backend's `.env` first, as `ALLOWED_ORIGINS=["http://localhost:3000"]`, and `FAULT_INJECTION=true` if you want the panel that simulates failures.

```sh
cd frontend
npm install
npm run dev          # http://localhost:3000, talks to ws://localhost:8787/ws
```

`NEXT_PUBLIC_WS_URL` points the page at another backend. It is read when the frontend is built.

### On a host

The backend needs no `.env` file. Give it the same settings as environment variables and run:

```sh
cd backend
uv sync --frozen --no-dev
uv run --no-dev python -m supportagent.cli.ingest     # migrations and the help center, safe to repeat
uv run --no-dev uvicorn --factory supportagent.api.app:create_app --host 0.0.0.0 --port "$PORT" --ws-max-size 65536 --no-proxy-headers
```

Set at least `DATABASE_URL`, `OPENROUTER_API_KEY`, the model lines of `.env.example`, and for a public address:

```sh
ALLOWED_ORIGINS=["https://your-frontend.example"]
REQUIRE_ORIGIN=true        # refuse clients that send no Origin
CLIENT_ADDRESS_HEADER=x-real-ip   # the header in which the host's proxy reports the visitor
SESSION_TTL_HOURS=48
```

The repo's `Dockerfile` does the same in an image, and that image is what runs on Railway. There, `/health` answers, a page from the allowed origin gets a session, a client without an origin is refused, and two visitors from two addresses each get their own count of new conversations, whatever headers they send.

### Demo accounts

Each session gets a private copy of this data. A refund in one session does not exist in the next.

| Account | PIN | Customer | Plan | Try |
|---|---|---|---|---|
| 10482917 | 4721 | Maya Okafor, Larkspur Design Co | Pro, monthly | Refund INV-30417, which is 5 days old. INV-29880 is 35 days old and outside the window. |
| 20573846 | 9034 | Daniel Reyes, Quillon Freight | Pro, annual | Refund INV-30388. At $1,440 it is above the $500 limit and goes to billing. |
| 31764025 | 1586 | Priya Natarajan, Brindle & Wye Accounting | Pro, monthly | Charged twice for one period, INV-30212 and INV-30213 |
| 53920164 | 2458 | Elena Sokolova, Harrowgate Robotics | Enterprise | INV-30390 is a $900 add-on and refundable. INV-30402 is $4,500 and above the limit. |
| 64037291 | 8815 | Marcus Bell, Ostrander Health Systems | Enterprise, suspended | Nothing is refunded or changed in chat |
| 75148302 | 3390 | Jun Watanabe, Pellam Studio | Free | No refunds on this plan |

Six more accounts are in `backend/src/supportagent/crm/fixtures.py`.

## The chat UI

The page is a fold over the event stream. `lib/state.ts` is one reducer that turns events into turns, steps and metrics, and every part of the page reads its result. Stored events replay through the same reducer, so a reload shows the conversation exactly as it happened, steps and traces included.

- Step indicators. Each tool call is a row: a spinner and "Searching the knowledge base…" while it runs, then its result and duration. Retries, a switch to the backup model and the grounding check are rows too.
- The answer streams as it is written. Text the model wrote before a tool call is dropped, and a withdrawn answer is replaced by the hand-off with a note that says so.
- A reply in progress can be stopped and resumed. A failed turn offers Try again, which picks up after the last finished step.
- The connection reconnects by itself and asks for the events it missed.
- The observability drawer reads the same state: no second data path, so what it shows cannot disagree with the chat.

The look follows a design file the project owner supplied: a dark, flat, monochrome system with one action colour. Its typefaces are not licensed for this repo, so Inter stands in, as the design file suggests.

Checked in a browser against the live backend: the full chain of identity check, policy search, account read and refund, a hand-off, an injected rate limit, a failing search, a switch to the backup model, a failed turn and its retry, stop and resume, a dropped connection, a reload, and a phone-sized screen. The reducer has nine tests.

## Evals

```sh
uv run python -m supportagent.evals.retrieval ../evals/retrieval_dev.jsonl ../evals/retrieval_heldout.jsonl
uv run python -m supportagent.evals.judge --runs 3
uv run python -m supportagent.evals.suite
uv run python -m supportagent.evals.bakeoff ../evals/candidates.hosted.json
```

- `retrieval` measures ranking and the scope gate on labeled questions. One set of 68 questions was written alongside the documents. Outside reviewers wrote a second set of 42.
- `judge` runs the grounding judge on 44 replies whose right verdict is known. Half are correct, half carry one seeded error.
- `suite` runs 55 scripted conversations against the configured model and checks which tools ran, what the database holds afterwards and what the agent said.
- `bakeoff` runs the suite for several candidate models and picks one by a rule fixed in advance: at least 95% of scenarios, then the fastest tool chain, then the lower cost. Every candidate runs without a fallback, is graded by another model, and gets the app's default time limits whatever the environment says. The speed that is compared leaves out the grounding check, because another model runs it.

### Results so far

Hosted models through OpenRouter, measured on 2026-11-26.

| Check | Result |
|---|---|
| Tests | 320 pass |
| Bake-off, 55 conversations run 3 times | Gemma 4 31B passed 164 of 165 with a median tool chain of 2.0 s at $0.0005 a turn, and is the agent model. Gemini 3.6 Flash passed 156 and GPT-6 Luna 155. |
| Grounding judge, Gemini 3.6 Flash, 44 replies run 3 times | 0 misses on 66 seeded errors, 1 false alarm on 66 correct replies |
| Retrieval, Cohere rerank-5-pro | The right section is first for 69 of 76 answerable questions and in the top 5 for 74 |
| Scope gate, 110 questions | 1 of 76 answerable questions escalated. 0 of 9 off-topic questions accepted. 2 of 22 near misses pass the threshold and are held for the grounding check. |

Read these with care.

- Qwen3.8-27B on Cerebras passed 15 of 15 screening conversations with the fastest tool chain, 1.6 s, in an earlier run. In the full bake-off Cerebras answered with rate limits at three conversations at once, so it was not compared. It may be the better model at a higher rate limit.
- Claude Haiku 4.5 and GLM-5.2 passed 13 and 12 of 15 in screening and did not reach the full run.
- The scenarios, the judge cases and the prompts have one author. The gate thresholds were tuned on the same 110 questions they are reported on, and two suite checks were loosened after a first bake-off run showed them failing correct answers.
- Nine off-topic questions are few.

## WebSocket protocol

One socket per client, at `/ws`. The client speaks first.

| Client sends | For |
|---|---|
| `hello` | Opens a new session or resumes one. Carries the last event number the client saw. |
| `user.message` | Starts a turn |
| `turn.retry` | Resumes a failed turn |
| `turn.cancel` | Stops the running turn |
| `fault.inject` | Queues a simulated failure, when the server allows it |

The server answers with events, each with a sequence number:

- session: `session.ready`, `session.state`, `error`
- turn: `turn.started`, `turn.finished`, `turn.failed`, `turn.interrupted`
- model: `llm.started`, `llm.delta`, `llm.reset`, `llm.retry`, `llm.fallback`, `llm.finished`, `llm.trace`
- tools: `tool.started`, `tool.retry`, `tool.error`, `tool.finished`
- guardrails: `retrieval.result`, `guardrail.verdict`, `answer.retracted`, `escalation.created`
- reply: `assistant.message`

`frontend/src/lib/protocol.ts` holds the TypeScript types. `python -m supportagent.cli.gen_types` writes that file from the pydantic models in `events.py` and `api/protocol.py`, and a test fails when it is stale.

## Layout

```text
backend/src/supportagent/
  agent/      the graph, the six tools, the tool guard, the judge, prompts, tickets
  api/        FastAPI app, session hub, client messages
  crm/        mock customers, policy constants, queries, per-session worlds
  kb/         chunker, ingest, hybrid retrieval, local and hosted embedders and rerankers
  llm/        one client for every model call: retries, budget, deadline, fallback, faults, cost
  evals/      retrieval, judge, scenario suite, bake-off
  cli/        preflight, ingest, chat, ws_chat, gen_types, gateway_check
  sql/        forward-only migrations
  redact.py   keeps PINs out of what is stored and sent
  session.py  runs one turn and reports it as events
  eventlog.py numbers a session's events and stores them
backend/tests/   the tests
kb/              16 help center documents, 96 chunks after ingest
evals/           question sets and bake-off candidate files
frontend/src/
  app/           layout, the one page, design tokens in globals.css
  components/    conversation, one turn, composer, top bar, observability drawer
  lib/state.ts   the reducer that folds the event stream into what the page shows
  lib/session.ts the WebSocket client: resume, reconnect, retry, cancel
  lib/protocol.ts generated from the backend's models
scripts/         dev database and dev server
```

## What comes next

1. Phase 1 is done. Open from it: a rerun of Qwen3.8-27B at a rate limit that allows it.
2. Phase 2 is deployed. Open from it: deploys are started by hand, not by a push.
3. Phase 3: voice through the same session, with Deepgram Flux for speech to text and text to speech.
4. Phase 4: hardening and the write-up.

## Known gaps

- The limit on new conversations is counted per address in one process's memory. Several backend processes would each count for themselves, and a restart forgets the counts.
- A browser is checked by an origin allowlist. A client that is not a browser can send any origin, so the limits above and a spending cap on the model key are what bound the cost.
- The wrong-PIN lockout counts per conversation. A new conversation starts a new count.
- The local reranker does not hold the scope gate. Use the hosted one, as the example settings do.
- The CRM is a set of fixtures. Nothing here talks to a real billing system.
