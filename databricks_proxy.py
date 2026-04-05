import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# --- Configuration ---

_SCRIPT_DIR = Path(__file__).parent

_DEFAULT_MODELS = [
    "databricks-gpt-5-4-mini",
    "databricks-gpt-5-4-nano",
    "databricks-claude-sonnet-4-6",
    "databricks-gpt-5-3-codex",
    "databricks-claude-opus-4-6",
    "databricks-gpt-5-4",
    "databricks-gpt-5-1-codex-max",
    "databricks-gpt-5-1-codex-mini",
    "databricks-gpt-5-2-codex",
    "databricks-claude-haiku-4-5",
    "databricks-gpt-5-2",
    "databricks-claude-opus-4-5",
    "databricks-gpt-5-1",
    "databricks-gemini-2-5-flash",
    "databricks-gemini-2-5-pro",
    "databricks-gpt-5",
    "databricks-gpt-5-mini",
    "databricks-gpt-5-nano",
    "databricks-claude-sonnet-4-5",
    "databricks-claude-opus-4-1",
    "databricks-gpt-oss-120b",
    "databricks-gpt-oss-20b",
    "databricks-claude-sonnet-4",
    "databricks-claude-3-7-sonnet",
    "databricks-bge-large-en",
    "databricks-gemma-3-12b",
    "databricks-gte-large-en",
    "databricks-llama-4-maverick",
    "databricks-meta-llama-3-1-8b-instruct",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-qwen3-embedding-0-6b",
    "databricks-qwen3-next-80b-a3b-instruct",
]


def load_models() -> list[str]:
    models_file = _SCRIPT_DIR / "models.json"
    if models_file.exists():
        try:
            data = json.loads(models_file.read_text())
            if isinstance(data, list) and data:
                return data
            print(f"Warning: {models_file} is empty or malformed; using built-in model list.", flush=True)
        except Exception as e:
            print(f"Warning: Could not read {models_file}: {e}; using built-in model list.", flush=True)
    return _DEFAULT_MODELS


MODELS = load_models()
_MODELS_SOURCE = "models.json" if (_SCRIPT_DIR / "models.json").exists() else "built-in list"


class ProxyHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        data = [{"id": m} for m in MODELS]
        self.wfile.write(json.dumps({"object": "list", "data": data}).encode("utf-8"))

    def do_POST(self):
        print(">>> Bridge: Request received...", flush=True)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        req_model = "unknown-model"

        try:
            payload = json.loads(body.decode("utf-8"))

            # Capture the requested model to inject it back into response chunks later
            req_model = payload.get("model", "databricks-claude-sonnet-4-6")

            if "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")

            # Strip keys known to cause 400 errors on Databricks
            for key in ["parallel_tool_calls", "stream_options", "store", "metadata", "logprobs", "top_logprobs"]:
                payload.pop(key, None)

            # Strip response_format when streaming — Databricks rejects structured output + stream
            if payload.get("stream") is True:
                payload.pop("response_format", None)

            body = json.dumps(payload).encode("utf-8")
        except Exception:
            pass

        req = urllib.request.Request(DATABRICKS_URL, data=body, method="POST")
        for key, value in self.headers.items():
            if key.lower() not in ["host", "connection", "content-length", "accept-encoding"]:
                req.add_header(key, value)
        req.add_header("Content-Length", str(len(body)))
        req.add_header("Accept-Encoding", "identity")

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                print(f"<<< Bridge: Streaming {response.status}...", flush=True)
                self.send_response(response.status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                req_id = f"chatcmpl-{uuid.uuid4()}"
                created_time = int(time.time())
                is_first = True

                while True:
                    line = response.readline()
                    if not line:
                        break

                    line_str = line.decode("utf-8", errors="replace").strip()

                    if line_str.startswith("data: "):
                        if line_str == "data: [DONE]":
                            self.wfile.write(b"data: [DONE]\n\n")
                        else:
                            try:
                                chunk = json.loads(line_str[6:])

                                # Inject missing OpenAI-standard fields so ForgeCode's Rust parser doesn't drop chunks
                                chunk.update({
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": req_model,
                                })

                                if "choices" in chunk:
                                    for choice in chunk["choices"]:
                                        choice["index"] = 0
                                        if "delta" not in choice:
                                            choice["delta"] = {}
                                        if is_first:
                                            choice["delta"]["role"] = "assistant"
                                            is_first = False
                                        if "content" not in choice["delta"] and choice.get("finish_reason") is None:
                                            choice["delta"]["content"] = ""

                                # Strip usage — Databricks sends non-standard Anthropic extensions that crash ForgeCode
                                chunk.pop("usage", None)

                                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                            except Exception:
                                self.wfile.write(line_str.encode("utf-8") + b"\n\n")

                    self.wfile.flush()

                print("<<< Bridge: Done.", flush=True)

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            print(f"!!! Bridge Error: {e}", flush=True)
            self.send_response(500)
            self.end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForgeCode ↔ Databricks AI Gateway Bridge")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    DATABRICKS_URL = os.environ.get("DATABRICKS_AI_GATEWAY_URL")
    if not DATABRICKS_URL:
        print("Error: DATABRICKS_AI_GATEWAY_URL is not set.\n", file=sys.stderr)
        print("  Set it in your shell:", file=sys.stderr)
        print("    export DATABRICKS_AI_GATEWAY_URL=https://<workspace>.ai-gateway.azuredatabricks.net/mlflow/v1/chat/completions\n", file=sys.stderr)
        print("  Or copy .env.example to .env, fill in your URL, and source it:", file=sys.stderr)
        print("    cp .env.example .env && source .env\n", file=sys.stderr)
        sys.exit(1)

    print(f"""
  ForgeCode ↔ Databricks AI Gateway Bridge
  ─────────────────────────────────────────
  Listening on : http://{args.host}:{args.port}
  Forwarding to: {DATABRICKS_URL}
  Models       : {len(MODELS)} available (from {_MODELS_SOURCE})

  Configure ForgeCode (first time only):
    forge provider login openai_compatible
      URL     → http://{args.host}:{args.port}
      API Key → <Your Databricks Personal Access Token>

  Set active model:
    forge config set model databricks-claude-sonnet-4-6

  Press Ctrl+C to stop.
""", flush=True)

    server = HTTPServer((args.host, args.port), ProxyHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBridge stopped. Goodbye!", flush=True)
        server.server_close()
        sys.exit(0)
