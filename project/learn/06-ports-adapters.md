# 06 — Ports and adapters: plugging `csv_analyst` into Geekbot (step C2)

## 1. The problem C2 solves
`csv_analyst` talks to a model through one small interface, and Geekbot already has its own way of talking to Ollama (settings, `keep_alive`, `num_ctx`, timeouts, logging). C2 connects the two **without editing the package's logic** and without a second, hidden set of Ollama settings.

Two ideas do the work: a **port** (an interface the package defines) and an **adapter** (a class on Geekbot's side that fits it).

## 2. Ports and adapters (hexagonal architecture)
- **Port:** the interface the inner code needs. Here it is `LLMProvider` in `src/csv_analyst/llm/base.py`: one method, `generate_json(prompt, *, system, temperature, seed, schema) -> str`. The pipeline calls only this.
- **Adapter:** a class that fits the port and hides the outside world behind it. The package ships one (`OllamaProvider`, which uses `requests` and its own settings). Geekbot writes another: `geekbot.csv_mode.provider`, which uses Geekbot's settings and the same HTTP library as the rest of Geekbot (`httpx`).
- **Why bother:** the inner code never learns which backend, which timeouts or which logging are behind the port. You can swap the adapter (or use a fake one in a test) without touching the pipeline.
- **Trade-off:** one more layer to read, and a rule to keep (only the adapter knows Ollama). For one backend that is a small cost, but here it also gives us a clean place to enforce Geekbot's rules (D-041: nothing leaves the machine).

## 3. `Protocol`: structural typing
`LLMProvider` is a `typing.Protocol`. A class matches it **by shape**: if it has a method with the right name and signature, it fits. It does not have to inherit from `LLMProvider` or even import it.
- **Versus an abstract base class:** an ABC forces `class MyProvider(LLMProvider)`. A Protocol doesn't, so our adapter can live in Geekbot and the package never needs to know it exists.
- **Who checks it:** mypy, where the adapter is handed to the package (`CsvAnalystSession(..., llm=provider)`). That call is in `geekbot.csv_mode`, which stays under `strict = true`. The package itself has `ignore_errors = true` (C1's typing debt), but its annotations are still read, so a wrong signature in our adapter would be reported on our side.
- **Pitfall:** a Protocol is not checked at runtime unless marked `@runtime_checkable`, and even then only for method names. The test suite has to prove the behaviour.

## 4. Dependency injection through a parameter
`CsvAnalystSession.__init__` takes `llm: Optional[LLMProvider] = None`. If it is given, the session uses it; if not, it builds the package's own Ollama provider.
- That is **dependency injection**: the caller decides what the object talks to. No framework needed, just a constructor argument.
- It has a second payoff in tests: pass a fake provider that returns canned JSON, and the whole pipeline runs without Ollama.
- **Gotcha, read from the code:** the session still reads `model` separately (`model or settings.llm_model_main`) and uses that name in its `RunRecord` and for the time limit. So Geekbot must pass **both** `llm=` and `model=`, with the same name, or the record would say the wrong model.

## 5. Two settings systems in one process
The package has its own `pydantic-settings` object (`csv_analyst.config.settings`, prefix `CSV_ANALYST_`, created when the module is imported). Geekbot has `GEEKBOT_` (S0.2).
- **Both read `.env`** in the working directory, and both use `extra="ignore"`, so each ignores the other's variables. No crash, but also no warning about a typo.
- **Rule for C2:** Geekbot owns what touches Ollama and the machine (host, model, `num_ctx`, `keep_alive`, timeouts). The package's prompt and pipeline settings (samples, retries, prompt versions) stay at the defaults Gigalogy chose by measurement; changing them means re-running the benchmark.
- **What still reads the package's settings:** the per-question time limit. `pipeline.run_pipeline` takes a `time_limit_s` argument, but `CsvAnalystSession.ask` does not pass one, so the default applies: 600 s for the main model and 300 s for the `qwen2.5-coder:3b` "fast" tier (matched by model name). Two facts matter for the design:
  - the limit is checked **before each attempt or sample batch**; a model call already running finishes first, so the real worst case is the limit plus one call;
  - how Geekbot sets its own cap is a design choice for C2 (see the step log, once the design is approved).

## 6. Ollama JSON mode and structured output
The package calls `POST /api/generate` with `"stream": false`, and a `format` field:
- `"format": "json"` only forces the output to **be** JSON. The model can still return valid JSON of the wrong shape (the package saw a truncated `code` string this way).
- `"format": <JSON Schema object>` makes Ollama decode **to that shape** (constrained decoding). The package's comments say this needs Ollama 0.32+; Geekbot's measured Ollama version is 0.34.2 (P0 notes).
- `generate_json` takes an optional `schema`; an adapter that can't use it may fall back to plain JSON mode, because the planner has a one-shot repair path.
- **The contract:** return the raw text. A provider must **not** repair malformed JSON; the caller parses and validates, and counts the failure.

## 7. What an adapter must get right (the C2 checklist)
1. **Same payload shape as the original:** `model`, `prompt`, `system`, `format` (schema or `"json"`), `stream: false`, and `options` with `temperature`, `num_ctx`, and `seed` only when given. Changing how the model is called changes the results Gigalogy measured.
2. **`num_ctx` and `keep_alive` from Geekbot's settings.** `num_ctx` must equal what warm-up uses, or Ollama reloads the model (lesson 03 §4). A long `keep_alive` keeps the model resident (R-2).
3. **The "prompt too long" guard, before sending.** Ollama silently keeps only the end of an over-long prompt, which would drop the schema and make the model answer from what is left. The package estimates tokens (3.0 characters per token, 512 reserved for the reply) and raises `PromptTooLongError`.
4. **Error mapping:** transport failures (timeout, connection refused, non-2xx, bad JSON body) become `LLMProviderError`. The pipeline treats that as one lost sample and, if every sample fails that way, reports a service error rather than blaming the data.
5. **Load events:** Ollama's response says how long loading the model took (`load_duration`). The plan wants that recorded so "no model loads during a query" (R-2) can be checked. `RunStats` does not exist in the code yet (it arrives with the graph), so the design must decide where it goes meanwhile.
6. **Empty `response`:** the package falls back to the `thinking` field when `response` is empty, for models that put their structured answer there. Geekbot keeps thinking off by default; whether to copy the fallback is a design question.

## 8. Testing an adapter without a server
S0.3 already used `httpx.MockTransport` (see `tests/unit/test_models_client.py`): the test hands `httpx.Client` a function that plays the server, so a test can check the exact request that was sent and make the "server" return success, a 500, a refused connection or garbage. No Ollama, no GPU, no network.
- **What to assert:** the URL and JSON body that left the adapter; the string that comes back on success; the exception type on each failure; that an over-long prompt raises **before** any request is made (the fake server is never called).
- **What a fake cannot tell you:** whether the real model follows the schema, or how long a call takes. That is the live `geekbot csv ask` run (marked `live`, uses the GPU).

## Interview-style questions
1. **What is the difference between a port and an adapter?** The port is the interface the inner code depends on; the adapter is an implementation of it for one concrete outside system. The inner code only sees the port.
2. **Why use a `Protocol` instead of an abstract base class here?** Structural typing lets Geekbot's class fit the package's interface without inheriting from it or being known to the package, so the vendored code stays unchanged.
3. **What does dependency injection buy you in this step?** The pipeline can run with Geekbot's provider in production and a fake in tests, with no change to the pipeline. The constructor argument is the whole mechanism.
4. **Why is `"format": "json"` weaker than passing a schema?** It guarantees syntactically valid JSON, not the right keys or types. A schema makes the decoder produce that shape.
5. **Why must `num_ctx` match between warm-up and the real calls?** Ollama treats a different context size as a different load and reloads the model, which breaks the "no model loads during a query" rule.
6. **The package has its own settings object. How do you stop two settings systems from fighting?** Give them different env prefixes, let Geekbot own everything that touches the machine, pass those values in explicitly, and leave the measured defaults of the rest alone.
