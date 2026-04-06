# ForgeCode - Databricks AI Gateway Bridge: Agents Guide

## Context & Objective
This repository contains a Python proxy that sits between [ForgeCode CLI](https://forgecode.dev) and an [Azure Databricks AI Gateway](https://azure.microsoft.com/en-us/products/databricks/).

**The Objective:** Connect ForgeCode to Databricks using the `openai_compatible` provider. Because Databricks enforces strict payload validation and returns non-standard SSE (Server-Sent Events) chunks, a direct connection fails. The proxy intercepts, cleans, and restructures the HTTP traffic to make both sides happy.

## Configuration

The proxy uses these configuration sources, in order of precedence:

| Setting | Source | Details |
|---|---|---|
| Databricks endpoint URL | `DATABRICKS_AI_GATEWAY_URL` env var | **Required.** Use the AI Gateway base URL; full endpoint URLs are normalized automatically. |
| Available models and endpoint mapping | `models.json` in the repo root | Optional. Supports model IDs plus per-model endpoint alias mapping (`cursor`, `openai`, `mlflow`). |
| Listen port | `--port` CLI argument | Default: `8080` |
| Bind address | `--host` CLI argument | Default: `127.0.0.1` |

Set `DATABRICKS_AI_GATEWAY_URL` permanently in your shell profile (`~/.zshrc` or `~/.bash_profile`), or use a `.env` file (gitignored):
```bash
cp .env.example .env   # fill in your URL, then: source .env
```


## The Architecture of the Proxy

The proxy (`databricks_proxy.py`) handles three major architectural incompatibilities:

1. **The Validation Block (`GET /models`)**
   - **Issue:** ForgeCode sends `GET /models` to validate the endpoint before starting any chat session. Databricks returns a `404 Not Found`, crashing ForgeCode immediately.
   - **Fix:** The proxy mocks `GET /models` by returning a JSON list of 32 known Databricks model IDs (e.g., `databricks-claude-sonnet-4-6`).
2. **The Payload Block (`POST /chat/completions`)**
   - **Issue:** ForgeCode automatically includes `parallel_tool_calls`, `stream_options`, `store`, and `metadata`. When ForgeCode requests `stream: true`, it occasionally also forces `response_format: {"type": "json_object"}`. Databricks strict validation rejects these with a `400 Bad Request` ("Structured output is not currently supported with streaming").
   - **Fix:** The proxy intercepts the JSON payload, gracefully deletes these incompatible keys, and forwards the cleaned request to the Databricks API.
3. **The Streaming Block (SSE Deserialization)**
   - **Issue:** Databricks streams the text back, but it omits several standard OpenAI metadata fields (like `id`, `object`, `created`, `index`, and `model`). ForgeCode relies on `reqwest-eventsource` which uses strict Rust structs. If these fields are missing, ForgeCode silently drops the chunks or hangs indefinitely on "Synthesizing" because it assumes the packet is malformed.
   - **Fix:** The proxy parses every incoming `data: {...}` line from Databricks. It injects a UUID for `id`, sets `object: chat.completion.chunk`, adds a Unix timestamp for `created`, and ensures `delta.content` exists if `finish_reason` is null. It also aggressively strips the `usage` object from Databricks because it contains non-standard Anthropic extensions that crash ForgeCode. Finally, it ensures perfect `\n\n` framing and a clean `Connection: close` on the final `[DONE]` signal.

## What We Tried That Did Not Work

If you are modifying this project, **do not attempt the following**, as we have already proven they fail, at least as recently as ForgeCode v2.6.0:

- **Bypassing validation via `.forge.toml`:** We tried manually setting `model_id` and skipping the `models` endpoint. ForgeCode's internal logic *still* tries to fetch the models list and fails.
- **Using the Native OpenAI Provider:** We tried setting `OPENAI_BASE_URL` and `OPENAI_API_BASE` and using ForgeCode's native `OpenAI` provider (which skips the `/models` validation check). This failed because ForgeCode hardcodes `api.openai.com` in the compiled binary; it completely ignores environment variables.
- **Using `Custom Providers` in `forge.toml`:** We followed the official documentation to create a `[[providers]]` block pointing directly to Databricks. While this solved the `GET /models` crash (by letting us specify a local `models.json` file), it failed on the `400 Bad Request`. There is no configuration option in ForgeCode to disable `parallel_tool_calls` or `response_format`, meaning direct POST requests to Databricks will always be rejected.

## Next Steps for the Current Session

**Completed:**
- ✅ `DATABRICKS_URL` reads from `DATABRICKS_AI_GATEWAY_URL` environment variable; proxy exits with a clear error if unset
- ✅ Hardcoded workspace URL and any personal data removed from source code
- ✅ Models list extracted to `models.json` (editable, version-controlled, no secrets); built-in fallback list retained
- ✅ `--port` and `--host` CLI arguments added via `argparse`
- ✅ Startup message shows full configuration instructions (ForgeCode login command, model selection)
- ✅ Graceful `KeyboardInterrupt` / Ctrl+C handling — no more debug tracebacks on exit
- ✅ Python best practices: bare `except:` replaced with `except Exception:`, f-strings, consistent quote style, `pathlib` for file paths

**Remaining tasks for the AI agent reading this:**
1. Add tests — at minimum: unit tests for payload sanitization logic and SSE chunk enrichment; optionally integration tests using a mock HTTP server
2. Consider adding a `--models-file` CLI argument to allow specifying a custom models file path
3. Investigate whether the Databricks AI Gateway exposes an API endpoint to dynamically fetch available model IDs (would eliminate the need to maintain `models.json` manually)
