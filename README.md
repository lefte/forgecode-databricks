# ForgeCode-Databricks AI Gateway Bridge

## Overview
This project is a lightweight, zero-dependency Python proxy designed to bridge [ForgeCode CLI](https://forgecode.dev) with the [Azure Databricks AI Gateway](https://azure.microsoft.com/en-us/products/databricks/). 

Because ForgeCode is heavily optimized for standard OpenAI-compatible endpoints and Databricks implements a strict, slightly modified version of the specification, direct communication between the two systems fails. This middleware script sits between them, intercepting and translating requests and responses in real-time to ensure seamless compatibility, including streaming text generation and tool calling.

## The Problem
If you attempt to connect ForgeCode directly to a Databricks AI Gateway endpoint, you will encounter three critical blockers:

1. **The `/models` Endpoint Missing:** ForgeCode validates models upon initialization by calling `GET /models`. Databricks does not implement this route, returning a `404 Not Found`, which crashes ForgeCode's startup sequence.
2. **Strict Payload Validation:** ForgeCode sends advanced parameters like `parallel_tool_calls` and `response_format: {"type": "json_object"}`. When combined with `stream: true`, Databricks rejects these with a `400 Bad Request` ("Structured output is not currently supported with streaming").
3. **Incomplete SSE Streams:** Databricks returns a minimal Server-Sent Events (SSE) stream. ForgeCode's strict Rust deserializer expects standard OpenAI chunk metadata (`id`, `object`, `created`, `index`, etc.). Without these, ForgeCode silently drops the chunks or hangs indefinitely on "Synthesizing".

## The Solution
This Python proxy (`databricks_proxy.py`) acts as a transparent middleware layer:

- **Mocks the `/models` Route:** It returns a list of available Databricks models (see step two below) allowing ForgeCode to initialize successfully.
- **Sanitizes Outgoing Requests:** It intercepts `POST /chat/completions`, stripping out the incompatible parameters (`parallel_tool_calls`, `response_format`, `stream_options`) before forwarding the payload to Databricks.
- **Routes Per Model Family:** It maps each model to the correct Databricks endpoint path (for example `databricks-gpt*` → `/cursor/v1/chat/completions`, `databricks-claude*` → `/mlflow/v1/chat/completions`) and forwards each request accordingly.
- **Adapts Chat ↔ Responses Payloads for GPT Cursor Models:** Some cursor-backed GPT models enforce Responses API semantics (`input`, `max_output_tokens`) even on chat-completions routes. The proxy converts request/response shapes so chat-style clients keep working.
- **Enriches Incoming Streams:** It parses the SSE stream returning from Databricks, injects the missing OpenAI metadata into every chunk, ensures perfect `\n\n` framing, and properly handles the `[DONE]` signal so ForgeCode can render the live stream flawlessly.
- **Normalizes Structured `delta.content`:** Some Databricks/Claude streaming chunks emit structured arrays (for reasoning summaries) where strict clients expect a plain string. The proxy now coerces non-string `delta.content` into text so OpenAI-compatible client validators do not fail.

## Setup

### 1. Set your Databricks AI Gateway URL

The proxy requires one environment variable: your Databricks AI Gateway base URL. It will derive the host and then append the model-specific endpoint path from `models.json`.

**Option A — Shell export (current session only):**

```bash
export DATABRICKS_AI_GATEWAY_URL="https://<workspace-id>.ai-gateway.azuredatabricks.net"
```

**Option B — Persist it in your shell profile (recommended):**

```bash
# Add to ~/.zshrc or ~/.bash_profile, then restart your terminal or run `source ~/.zshrc`
export DATABRICKS_AI_GATEWAY_URL="https://<workspace-id>.ai-gateway.azuredatabricks.net"
```

**Option C — `.env` file:** *(Useful for sharing with coworkers, or Mom)*

```bash
cp .env.example .env
# Edit .env and fill in your URL, then:
source .env
```

> Your endpoint URL is found in the Databricks workspace under **AI Gateway → your endpoint → View endpoint details**.
> Existing full endpoint values (for example `/mlflow/v1/chat/completions`) are still accepted; the proxy will normalize them to the gateway base URL.

### 2. (Optional) Customize model-to-endpoint mapping

Edit `models.json` to match the models enabled in your AI Gateway and assign each to an endpoint alias (`cursor`, `mlflow`, or `openai`). The proxy loads this file automatically on startup; if it's missing, a built-in default list and endpoint rules are used. The bridge also auto-adapts chat-completions payloads to Responses-style fields when required by cursor/openai-backed models. For periodic updates to this list, I have browsed to the Databricks AI Gateway dashboard and copied the table of models to a file, then used the following prompt to help populate the models JSON file:

```bash
I have created a new markdown file @[databricks-models.md] that contains the most-current updated list of Databricks models copied from the website. Scrape this file for the new model names and update the @[models.json] file with the appropriate new and updated models. Use the same endpoint logic as before (gpt models use cursor, anthropic models use mlflow) I think the new models in markdown can be read using a 4n-3 formula, where the model names are on lines 1, 5, 9, 13, 17, 21, and so on.
```

```json
{
  "endpoints": {
    "mlflow": "/mlflow/v1/chat/completions",
    "cursor": "/cursor/v1/chat/completions",
    "openai": "/openai/v1/responses"
  },
  "models": [
    {"id": "databricks-claude-haiku-4-5", "endpoint": "mlflow"},
    {"id": "databricks-gpt-5-3-codex", "endpoint": "cursor"}
  ]
}
```

### 3. Run the proxy

```bash
python3 databricks_proxy.py
# Optional flags:
#   --port 9090       bind to a different port (default: 8080)
#   --host 0.0.0.0    bind to all interfaces (default: 127.0.0.1)
#   --upstream-timeout 300   upstream read timeout seconds before keep-alive logic
#   --max-read-timeouts 24   consecutive timeouts allowed before ending stream
```

### 4. Configure ForgeCode (First time only)

Your variables will be output by the proxy based on what you pass in. You only need to set these once in ForgeCode. You can create a Databricks token under personal `Settings > Developer > Manage Access Tokens`, but here are the defaults:

```bash
forge provider login openai_compatible
# URL:     http://127.0.0.1:8080
# API Key: <Your Databricks Personal Access Token>
```

### 5. Set your active model and start chatting

```bash
forge config set model databricks-claude-sonnet-4-6
# or
:model
```

You can verify your settings with `forge info` or `:info`.

## Troubleshooting

### `Unauthorized: Invalid access token`

If direct `curl` to Databricks works but proxy clients fail, the most common cause is client-side auth config drift:

- Keep one active PAT value everywhere (do not mix old/new keys across files or profiles).
- For OpenCode with `@ai-sdk/openai-compatible`, use `options.apiKey` (camelCase), not `apikey`.
- Enter raw token only (no `Bearer ` prefix in saved key fields).

### Type validation error: `choices[0].delta.content` expected string, received array

This happens when upstream emits structured reasoning blocks in stream chunks. The proxy now flattens those to string content before forwarding. Restart the proxy after pulling the latest changes.

### Bad Request: `Unsupported parameter: 'messages'`

Some cursor GPT models require Responses-style request fields (`input`, `max_output_tokens`) and reject `messages`. The proxy now auto-converts chat-style messages into Responses payloads for these models.

### Bad Request: `Invalid value: 'tool'` (`input[n]`)

This indicates tool-result turns were forwarded as chat role `tool` in a Responses request. The proxy now maps:

- assistant tool calls → `type: "function_call"` items
- tool messages → `type: "function_call_output"` items

to keep multi-step tool loops valid.

### Partial reply then early `[DONE]` (especially with tool-calling)

The stream parser now buffers complete SSE events (including multi-line `data:` payloads), translates `response.output_item.done` function calls into `delta.tool_calls`, and emits `finish_reason: "tool_calls"` when appropriate so clients continue execution instead of stopping early.

### `!!! Bridge Error: The read operation timed out`

Long-running requests can legitimately go quiet for extended periods. The proxy now sends SSE keep-alive comments on upstream read timeouts and only terminates after a configurable number of consecutive timeouts. Tune with:

- `--upstream-timeout` (seconds per upstream read wait)
- `--max-read-timeouts` (how many consecutive waits before ending the stream)

## Credential rotation checklist

Rotate any credential that appeared in terminal history, chat, logs, or local config during debugging:

1. Databricks PAT(s) used for this proxy/provider.
2. OpenCode provider API key entries for `forgecode-databricks` (replace with the new PAT everywhere it is stored).
3. Any other non-redacted API keys currently present in local OpenCode config/auth files (for example BrowserStack, xAI, Mistral, Ollama, MCP auth headers), if they were exposed in shared logs/chat.

*[F-D Bridge was built with the help of both Google Gemini and GitHub Copilot, with Erik Hanson in the architect seat.]*
