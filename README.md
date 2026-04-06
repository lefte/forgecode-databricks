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
- **Enriches Incoming Streams:** It parses the SSE stream returning from Databricks, injects the missing OpenAI metadata into every chunk, ensures perfect `\n\n` framing, and properly handles the `[DONE]` signal so ForgeCode can render the live stream flawlessly.

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

Edit `models.json` to match the models enabled in your AI Gateway and assign each to an endpoint alias (`cursor`, `mlflow`, or `openai`). The proxy loads this file automatically on startup; if it's missing, a built-in default list and endpoint rules are used. For chat-completions payloads (`messages`), the proxy automatically prefers `cursor` over `openai` when both exist, to avoid Responses API payload mismatch errors.

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

*[F-D Bridge was built with the help of both Google Gemini and GitHub Copilot, with Erik Hanson in the architect seat.]*
