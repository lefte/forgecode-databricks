import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

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


_DEFAULT_ENDPOINT_PATHS = {
    "mlflow": "/mlflow/v1/chat/completions",
    "cursor": "/cursor/v1/chat/completions",
    "openai": "/openai/v1/responses",
}


def infer_endpoint_alias(model_id: str) -> str:
    if model_id.startswith("databricks-gpt"):
        return "cursor"
    return "mlflow"


def normalize_endpoint_path(path: str) -> str:
    normalized = path.strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized


def parse_models_config(data: object) -> tuple[list[str], dict[str, str], dict[str, str]]:
    endpoint_paths = dict(_DEFAULT_ENDPOINT_PATHS)
    model_endpoint_aliases: dict[str, str] = {}

    if isinstance(data, dict):
        raw_endpoints = data.get("endpoints")
        if isinstance(raw_endpoints, dict):
            for alias, path in raw_endpoints.items():
                if isinstance(alias, str) and alias.strip() and isinstance(path, str) and path.strip():
                    endpoint_paths[alias.strip()] = normalize_endpoint_path(path)
        raw_models = data.get("models", [])
    else:
        raw_models = data

    if not isinstance(raw_models, list):
        raise ValueError("Model configuration must contain a list of models.")

    models: list[str] = []
    for entry in raw_models:
        model_id: str | None = None
        endpoint_alias: str | None = None

        if isinstance(entry, str):
            model_id = entry.strip()
        elif isinstance(entry, dict):
            raw_id = entry.get("id")
            if isinstance(raw_id, str):
                model_id = raw_id.strip()
            raw_endpoint = entry.get("endpoint")
            if isinstance(raw_endpoint, str):
                endpoint_alias = raw_endpoint.strip()

        if not model_id:
            continue

        if not endpoint_alias:
            endpoint_alias = infer_endpoint_alias(model_id)

        models.append(model_id)
        model_endpoint_aliases[model_id] = endpoint_alias

    if not models:
        raise ValueError("No valid models found in configuration.")

    return models, model_endpoint_aliases, endpoint_paths


def load_models() -> tuple[list[str], dict[str, str], dict[str, str], str]:
    models_file = _SCRIPT_DIR / "models.json"
    if models_file.exists():
        try:
            data = json.loads(models_file.read_text())
            models, model_endpoint_aliases, endpoint_paths = parse_models_config(data)
            return models, model_endpoint_aliases, endpoint_paths, "models.json"
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not read {models_file}: {e}; using built-in model list.", flush=True)
    models, model_endpoint_aliases, endpoint_paths = parse_models_config(_DEFAULT_MODELS)
    return models, model_endpoint_aliases, endpoint_paths, "built-in list"


MODELS, MODEL_ENDPOINT_ALIASES, ENDPOINT_PATHS, _MODELS_SOURCE = load_models()
DATABRICKS_BASE_URL = ""
UPSTREAM_TIMEOUT_SECONDS = 300
MAX_READ_TIMEOUTS = 24


def extract_gateway_base_url(configured_url: str) -> str:
    parsed = urlsplit(configured_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("DATABRICKS_AI_GATEWAY_URL must include scheme and host.")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _select_compatible_endpoint_alias(
    endpoint_alias: str,
    endpoint_paths: dict[str, str],
    payload: dict[str, object] | None,
) -> str:
    if payload is None:
        return endpoint_alias

    # ForgeCode sends chat-completions style payloads (messages). If a model is mapped to
    # responses, prefer Databricks' cursor chat endpoint when available.
    if "messages" in payload and endpoint_alias == "openai" and "cursor" in endpoint_paths:
        return "cursor"

    if "input" in payload and endpoint_alias == "cursor" and "openai" in endpoint_paths:
        return "openai"

    return endpoint_alias


def resolve_endpoint_path(
    model_id: str,
    model_endpoint_aliases: dict[str, str],
    endpoint_paths: dict[str, str],
    payload: dict[str, object] | None = None,
) -> str:
    endpoint_alias = model_endpoint_aliases.get(model_id, infer_endpoint_alias(model_id))
    endpoint_alias = _select_compatible_endpoint_alias(endpoint_alias, endpoint_paths, payload)
    endpoint_path = endpoint_paths.get(endpoint_alias)
    if endpoint_path is not None:
        return endpoint_path

    fallback_alias = infer_endpoint_alias(model_id)
    return endpoint_paths.get(fallback_alias, _DEFAULT_ENDPOINT_PATHS["mlflow"])


def resolve_target_url(endpoint_path: str) -> str:
    return urljoin(f"{DATABRICKS_BASE_URL}/ai-gateway/", endpoint_path.lstrip("/"))


def _as_responses_content(role: str, content: object) -> list[dict[str, str]]:
    content_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": content_type, "text": content}]

    if isinstance(content, list):
        converted: list[dict[str, str]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            text = part.get("text")
            if not isinstance(text, str):
                continue
            if part_type in {"input_text", "output_text"}:
                converted.append({"type": part_type, "text": text})
            elif part_type == "text":
                converted.append({"type": content_type, "text": text})
        return converted

    return []


def _coerce_message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        return "".join(text_parts)
    return ""


def _tool_call_to_response_item(tool_call: object) -> dict[str, object] | None:
    if not isinstance(tool_call, dict):
        return None
    if tool_call.get("type") != "function":
        return None

    function_obj = tool_call.get("function")
    if not isinstance(function_obj, dict):
        return None
    function_name = function_obj.get("name")
    if not isinstance(function_name, str) or not function_name:
        return None
    arguments = function_obj.get("arguments")
    if not isinstance(arguments, str):
        arguments = ""

    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{uuid.uuid4().hex}"

    return {
        "type": "function_call",
        "call_id": call_id,
        "name": function_name,
        "arguments": arguments,
    }


def convert_chat_to_responses_payload(payload: dict[str, object]) -> dict[str, object]:
    converted = dict(payload)
    raw_messages = converted.get("messages")
    if not isinstance(raw_messages, list):
        return converted

    input_items: list[dict[str, object]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not isinstance(role, str):
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            output_text = _coerce_message_content_to_text(message.get("content"))
            input_items.append({
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": output_text,
            })
            continue

        responses_content = _as_responses_content(role, message.get("content"))
        if responses_content:
            input_items.append({
                "role": role,
                "content": responses_content,
            })

        if role == "assistant":
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for tool_call in raw_tool_calls:
                    response_item = _tool_call_to_response_item(tool_call)
                    if response_item is not None:
                        input_items.append(response_item)

    converted["input"] = input_items
    converted.pop("messages", None)
    if "max_tokens" in converted and "max_output_tokens" not in converted:
        converted["max_output_tokens"] = converted.pop("max_tokens")

    return converted


def _normalize_tools_for_responses(payload: dict[str, object]) -> None:
    raw_tools = payload.get("tools")
    if not isinstance(raw_tools, list):
        return

    normalized_tools: list[object] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") != "function":
            normalized_tools.append(tool)
            continue

        if isinstance(tool.get("name"), str):
            normalized_tools.append(tool)
            continue

        function_obj = tool.get("function")
        if not isinstance(function_obj, dict):
            normalized_tools.append(tool)
            continue

        function_name = function_obj.get("name")
        if not isinstance(function_name, str) or not function_name:
            normalized_tools.append(tool)
            continue

        normalized_tool: dict[str, object] = {"type": "function", "name": function_name}
        if "description" in function_obj:
            normalized_tool["description"] = function_obj["description"]
        if "parameters" in function_obj:
            normalized_tool["parameters"] = function_obj["parameters"]
        if "strict" in function_obj:
            normalized_tool["strict"] = function_obj["strict"]
        normalized_tools.append(normalized_tool)

    payload["tools"] = normalized_tools


def _normalize_tool_choice_for_responses(payload: dict[str, object]) -> None:
    tool_choice = payload.get("tool_choice")
    if not isinstance(tool_choice, dict):
        return
    if tool_choice.get("type") != "function" or isinstance(tool_choice.get("name"), str):
        return

    function_obj = tool_choice.get("function")
    if not isinstance(function_obj, dict):
        return
    function_name = function_obj.get("name")
    if not isinstance(function_name, str) or not function_name:
        return

    payload["tool_choice"] = {"type": "function", "name": function_name}


def adapt_payload_for_endpoint(
    payload: dict[str, object],
    endpoint_path: str,
    endpoint_paths: dict[str, str],
) -> dict[str, object]:
    adapted = dict(payload)
    openai_endpoint_path = endpoint_paths.get("openai", _DEFAULT_ENDPOINT_PATHS["openai"])
    cursor_endpoint_path = endpoint_paths.get("cursor", _DEFAULT_ENDPOINT_PATHS["cursor"])
    if endpoint_path in {openai_endpoint_path, cursor_endpoint_path}:
        if "messages" in adapted and "input" not in adapted:
            adapted = convert_chat_to_responses_payload(adapted)
        if "max_tokens" in adapted and "max_output_tokens" not in adapted:
            adapted["max_output_tokens"] = adapted.pop("max_tokens")
        _normalize_tools_for_responses(adapted)
        _normalize_tool_choice_for_responses(adapted)

    return adapted


def _extract_text_from_response_output_item(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    content = item.get("content")
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for content_part in content:
        if not isinstance(content_part, dict):
            continue
        text = content_part.get("text")
        if isinstance(text, str):
            text_parts.append(text)
            continue
        summary = content_part.get("summary")
        if not isinstance(summary, list):
            continue
        for summary_item in summary:
            if not isinstance(summary_item, dict):
                continue
            summary_text = summary_item.get("text")
            if isinstance(summary_text, str):
                text_parts.append(summary_text)
    return "".join(text_parts)


def _build_chat_completion_chunk(
    req_id: str,
    created_time: int,
    req_model: str,
    delta: dict[str, object] | None = None,
    finish_reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": req_model,
        "choices": [{
            "index": 0,
            "delta": delta or {},
            "finish_reason": finish_reason,
        }],
    }


def _build_tool_call_delta_chunk(
    req_id: str,
    created_time: int,
    req_model: str,
    tool_call_index: int,
    call_id: str,
    function_name: str,
    arguments: str,
    include_role: bool,
) -> dict[str, object]:
    delta: dict[str, object] = {
        "tool_calls": [{
            "index": tool_call_index,
            "id": call_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": arguments,
            },
        }],
    }
    if include_role:
        delta["role"] = "assistant"
    return _build_chat_completion_chunk(
        req_id=req_id,
        created_time=created_time,
        req_model=req_model,
        delta=delta,
        finish_reason=None,
    )


def _coerce_delta_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return ""

    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue

        text = part.get("text")
        if isinstance(text, str):
            text_parts.append(text)
            continue

        # Databricks can stream Claude reasoning blocks as structured arrays.
        # Keep downstream chat chunk schema valid by flattening summary text.
        summary = part.get("summary")
        if not isinstance(summary, list):
            continue
        for summary_item in summary:
            if not isinstance(summary_item, dict):
                continue
            summary_text = summary_item.get("text")
            if isinstance(summary_text, str):
                text_parts.append(summary_text)

    return "".join(text_parts)


def _transform_upstream_chunk(
    raw_chunk: dict[str, object],
    req_id: str,
    created_time: int,
    req_model: str,
    needs_role_chunk: bool,
    stream_state: dict[str, object],
) -> tuple[list[dict[str, object]], bool]:
    transformed_chunks: list[dict[str, object]] = []

    raw_choices = raw_chunk.get("choices")
    if isinstance(raw_choices, list):
        chunk = dict(raw_chunk)
        chunk.update({
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": req_model,
        })
        for choice in raw_choices:
            if not isinstance(choice, dict):
                continue
            choice["index"] = 0
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
                choice["delta"] = delta
            if needs_role_chunk:
                delta["role"] = "assistant"
                needs_role_chunk = False
            if "content" in delta:
                delta["content"] = _coerce_delta_content_to_text(delta["content"])
            if "content" not in delta and choice.get("finish_reason") is None and "tool_calls" not in delta:
                delta["content"] = ""
        # Sanitize usage: keep only standard OpenAI fields to avoid crashing strict
        # clients on non-standard Anthropic extensions (cache_read_input_tokens, etc.).
        # Only attach usage on the final chunk (when finish_reason is set).
        raw_usage = raw_chunk.get("usage")
        has_finish_reason = any(
            isinstance(c, dict) and c.get("finish_reason") is not None
            for c in raw_choices
        )
        if isinstance(raw_usage, dict) and has_finish_reason:
            prompt_tokens = raw_usage.get("prompt_tokens")
            completion_tokens = raw_usage.get("completion_tokens")
            total_tokens = raw_usage.get("total_tokens")
            clean_usage: dict[str, object] = {}
            if isinstance(prompt_tokens, int):
                clean_usage["prompt_tokens"] = prompt_tokens
            if isinstance(completion_tokens, int):
                clean_usage["completion_tokens"] = completion_tokens
            if isinstance(total_tokens, int):
                clean_usage["total_tokens"] = total_tokens
            elif isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                clean_usage["total_tokens"] = prompt_tokens + completion_tokens
            if clean_usage:
                chunk["usage"] = clean_usage
            else:
                chunk.pop("usage", None)
        else:
            chunk.pop("usage", None)
        transformed_chunks.append(chunk)
        return transformed_chunks, needs_role_chunk

    event_type = raw_chunk.get("type")
    if not isinstance(event_type, str) or not event_type.startswith("response."):
        return transformed_chunks, needs_role_chunk

    if event_type == "response.output_text.delta":
        delta_text = raw_chunk.get("delta")
        if not isinstance(delta_text, str):
            delta_text = ""
        stream_state["saw_text_delta"] = True
        delta: dict[str, object] = {"content": delta_text}
        if needs_role_chunk:
            delta["role"] = "assistant"
            needs_role_chunk = False
        transformed_chunks.append(_build_chat_completion_chunk(
            req_id=req_id,
            created_time=created_time,
            req_model=req_model,
            delta=delta,
            finish_reason=None,
        ))
        return transformed_chunks, needs_role_chunk

    if event_type == "response.output_item.done":
        output_item = raw_chunk.get("item")
        if isinstance(output_item, dict) and output_item.get("type") == "function_call":
            call_id_value = output_item.get("call_id")
            if not isinstance(call_id_value, str) or not call_id_value:
                call_id_value = output_item.get("id") if isinstance(output_item.get("id"), str) else f"call_{uuid.uuid4().hex}"
            function_name = output_item.get("name")
            if not isinstance(function_name, str) or not function_name:
                function_name = "unknown_function"
            arguments = output_item.get("arguments")
            if not isinstance(arguments, str):
                arguments = ""

            tool_indices = stream_state.get("tool_call_indices")
            if not isinstance(tool_indices, dict):
                tool_indices = {}
                stream_state["tool_call_indices"] = tool_indices
            tool_call_index = tool_indices.get(call_id_value)
            if not isinstance(tool_call_index, int):
                next_tool_call_index = stream_state.get("next_tool_call_index", 0)
                tool_call_index = next_tool_call_index if isinstance(next_tool_call_index, int) else 0
                tool_indices[call_id_value] = tool_call_index
                stream_state["next_tool_call_index"] = tool_call_index + 1

            transformed_chunks.append(_build_tool_call_delta_chunk(
                req_id=req_id,
                created_time=created_time,
                req_model=req_model,
                tool_call_index=tool_call_index,
                call_id=call_id_value,
                function_name=function_name,
                arguments=arguments,
                include_role=needs_role_chunk,
            ))
            needs_role_chunk = False
            stream_state["saw_tool_call"] = True
            return transformed_chunks, needs_role_chunk

        output_text = _extract_text_from_response_output_item(output_item)
        saw_text_delta = bool(stream_state.get("saw_text_delta"))
        if output_text and not saw_text_delta:
            delta: dict[str, object] = {"content": output_text}
            if needs_role_chunk:
                delta["role"] = "assistant"
                needs_role_chunk = False
            transformed_chunks.append(_build_chat_completion_chunk(
                req_id=req_id,
                created_time=created_time,
                req_model=req_model,
                delta=delta,
                finish_reason=None,
            ))
            stream_state["emitted_text_from_output_item_done"] = True
        return transformed_chunks, needs_role_chunk

    if event_type == "response.completed":
        saw_tool_call = bool(stream_state.get("saw_tool_call"))
        saw_text_delta = bool(stream_state.get("saw_text_delta"))
        emitted_text_from_output_item_done = bool(stream_state.get("emitted_text_from_output_item_done"))
        if not saw_text_delta and not emitted_text_from_output_item_done:
            output = raw_chunk.get("response")
            if isinstance(output, dict):
                response_output = output.get("output")
                if isinstance(response_output, list):
                    combined_text = "".join(_extract_text_from_response_output_item(item) for item in response_output)
                    if combined_text:
                        delta: dict[str, object] = {"content": combined_text}
                        if needs_role_chunk:
                            delta["role"] = "assistant"
                            needs_role_chunk = False
                        transformed_chunks.append(_build_chat_completion_chunk(
                            req_id=req_id,
                            created_time=created_time,
                            req_model=req_model,
                            delta=delta,
                            finish_reason=None,
                        ))
        finish_chunk = _build_chat_completion_chunk(
            req_id=req_id,
            created_time=created_time,
            req_model=req_model,
            delta={},
            finish_reason="tool_calls" if saw_tool_call else "stop",
        )
        # Extract usage from response.completed payload if available.
        response_obj = raw_chunk.get("response")
        if isinstance(response_obj, dict):
            raw_usage = response_obj.get("usage")
            if isinstance(raw_usage, dict):
                prompt_tokens = raw_usage.get("input_tokens") or raw_usage.get("prompt_tokens")
                completion_tokens = raw_usage.get("output_tokens") or raw_usage.get("completion_tokens")
                clean_usage: dict[str, object] = {}
                if isinstance(prompt_tokens, int):
                    clean_usage["prompt_tokens"] = prompt_tokens
                if isinstance(completion_tokens, int):
                    clean_usage["completion_tokens"] = completion_tokens
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    clean_usage["total_tokens"] = prompt_tokens + completion_tokens
                if clean_usage:
                    finish_chunk["usage"] = clean_usage
        transformed_chunks.append(finish_chunk)
        return transformed_chunks, needs_role_chunk

    return transformed_chunks, needs_role_chunk


def sanitize_payload(payload: dict[str, object]) -> tuple[dict[str, object], str]:
    sanitized = dict(payload)
    req_model_value = sanitized.get("model")
    req_model = req_model_value if isinstance(req_model_value, str) and req_model_value else "databricks-claude-sonnet-4-6"

    if "max_completion_tokens" in sanitized:
        sanitized["max_tokens"] = sanitized.pop("max_completion_tokens")

    # Strip keys known to cause 400 errors on Databricks
    for key in ["parallel_tool_calls", "stream_options", "store", "metadata", "logprobs", "top_logprobs", "reasoning_effort"]:
        sanitized.pop(key, None)

    # Strip response_format when streaming — Databricks rejects structured output + stream
    if sanitized.get("stream") is True:
        sanitized.pop("response_format", None)

    return sanitized, req_model


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
        parsed_payload: dict[str, object] | None = None

        try:
            payload = json.loads(body.decode("utf-8"))
            if isinstance(payload, dict):
                payload, req_model = sanitize_payload(payload)
                parsed_payload = payload
                body = json.dumps(payload).encode("utf-8")
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

        endpoint_path = resolve_endpoint_path(req_model, MODEL_ENDPOINT_ALIASES, ENDPOINT_PATHS, parsed_payload)
        if parsed_payload is not None:
            forwarded_payload = adapt_payload_for_endpoint(parsed_payload, endpoint_path, ENDPOINT_PATHS)
            body = json.dumps(forwarded_payload).encode("utf-8")
            print(f">>> Bridge: Forward payload keys={sorted(forwarded_payload.keys())}", flush=True)
        target_url = resolve_target_url(endpoint_path)
        print(f">>> Bridge: Forwarding {req_model} to {target_url}", flush=True)
        req = urllib.request.Request(target_url, data=body, method="POST")
        for key, value in self.headers.items():
            if key.lower() not in ["host", "connection", "content-length", "accept-encoding"]:
                req.add_header(key, value)
        req.add_header("Content-Length", str(len(body)))
        req.add_header("Accept-Encoding", "identity")

        def write_to_client(chunk: bytes) -> bool:
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                print("<<< Bridge: Client disconnected.", flush=True)
                return False

        headers_sent = False
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_SECONDS) as response:
                print(f"<<< Bridge: Streaming {response.status}...", flush=True)
                try:
                    self.send_response(response.status)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    headers_sent = True
                except (BrokenPipeError, ConnectionResetError):
                    print("<<< Bridge: Client disconnected before headers.", flush=True)
                    return

                req_id = f"chatcmpl-{uuid.uuid4()}"
                created_time = int(time.time())
                needs_role_chunk = True
                stream_state: dict[str, object] = {}
                consecutive_read_timeouts = 0
                pending_event_lines: list[str] = []

                while True:
                    try:
                        line = response.readline()
                    except (TimeoutError, socket.timeout):
                        consecutive_read_timeouts += 1
                        print(
                            f"!!! Bridge: Upstream read timed out ({consecutive_read_timeouts}/{MAX_READ_TIMEOUTS}); keeping stream alive.",
                            flush=True,
                        )
                        if consecutive_read_timeouts >= MAX_READ_TIMEOUTS:
                            print("!!! Bridge: Max upstream read timeouts reached; ending stream.", flush=True)
                            if headers_sent:
                                write_to_client(b"data: [DONE]\n\n")
                            return
                        if headers_sent and not write_to_client(b": keep-alive\n\n"):
                            return
                        continue

                    consecutive_read_timeouts = 0
                    if not line:
                        # Flush any buffered SSE event if upstream closed without trailing blank line.
                        if pending_event_lines:
                            line_str = "\n".join(pending_event_lines)
                            pending_event_lines = []
                            if line_str == "[DONE]":
                                if not write_to_client(b"data: [DONE]\n\n"):
                                    return
                            else:
                                try:
                                    raw_chunk = json.loads(line_str)
                                    if isinstance(raw_chunk, dict):
                                        transformed_chunks, needs_role_chunk = _transform_upstream_chunk(
                                            raw_chunk=raw_chunk,
                                            req_id=req_id,
                                            created_time=created_time,
                                            req_model=req_model,
                                            needs_role_chunk=needs_role_chunk,
                                            stream_state=stream_state,
                                        )
                                        for transformed_chunk in transformed_chunks:
                                            if not write_to_client(f"data: {json.dumps(transformed_chunk)}\n\n".encode("utf-8")):
                                                return
                                except json.JSONDecodeError:
                                    pass
                        break

                    line_str = line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line_str == "":
                        if not pending_event_lines:
                            continue
                        event_payload = "\n".join(pending_event_lines)
                        pending_event_lines = []
                        if event_payload == "[DONE]":
                            if not write_to_client(b"data: [DONE]\n\n"):
                                return
                            continue
                        try:
                            raw_chunk = json.loads(event_payload)
                            if not isinstance(raw_chunk, dict):
                                continue
                            transformed_chunks, needs_role_chunk = _transform_upstream_chunk(
                                raw_chunk=raw_chunk,
                                req_id=req_id,
                                created_time=created_time,
                                req_model=req_model,
                                needs_role_chunk=needs_role_chunk,
                                stream_state=stream_state,
                            )
                            for transformed_chunk in transformed_chunks:
                                if not write_to_client(f"data: {json.dumps(transformed_chunk)}\n\n".encode("utf-8")):
                                    return
                        except json.JSONDecodeError:
                            continue
                        continue

                    if line_str.startswith(":"):
                        # Upstream keep-alive comment.
                        continue
                    if line_str.startswith("data:"):
                        payload_part = line_str[5:]
                        if payload_part.startswith(" "):
                            payload_part = payload_part[1:]
                        pending_event_lines.append(payload_part)
                        continue

                print("<<< Bridge: Done.", flush=True)

        except urllib.error.HTTPError as e:
            try:
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(e.read())
            except (BrokenPipeError, ConnectionResetError):
                print("<<< Bridge: Client disconnected before HTTP error response.", flush=True)
        except (BrokenPipeError, ConnectionResetError):
            print("<<< Bridge: Client disconnected during stream.", flush=True)
        except Exception as e:
            print(f"!!! Bridge Error: {e}", flush=True)
            try:
                if headers_sent:
                    write_to_client(b"data: [DONE]\n\n")
                else:
                    self.send_response(500)
                    self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForgeCode ↔ Databricks AI Gateway Bridge")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    parser.add_argument(
        "--upstream-timeout",
        type=int,
        default=UPSTREAM_TIMEOUT_SECONDS,
        help="Upstream connect/read timeout seconds before keep-alive logic engages (default: 300)",
    )
    parser.add_argument(
        "--max-read-timeouts",
        type=int,
        default=MAX_READ_TIMEOUTS,
        help="Number of consecutive upstream read timeouts before ending the stream (default: 24)",
    )
    args = parser.parse_args()
    UPSTREAM_TIMEOUT_SECONDS = max(1, args.upstream_timeout)
    MAX_READ_TIMEOUTS = max(1, args.max_read_timeouts)

    configured_gateway_url = os.environ.get("DATABRICKS_AI_GATEWAY_URL")
    if not configured_gateway_url:
        print("Error: DATABRICKS_AI_GATEWAY_URL is not set.\n", file=sys.stderr)
        print("  Set it in your shell:", file=sys.stderr)
        print("    export DATABRICKS_AI_GATEWAY_URL=https://<workspace>.ai-gateway.azuredatabricks.net\n", file=sys.stderr)
        print("  Or copy .env.example to .env, fill in your URL, and source it:", file=sys.stderr)
        print("    cp .env.example .env && source .env\n", file=sys.stderr)
        sys.exit(1)

    try:
        DATABRICKS_BASE_URL = extract_gateway_base_url(configured_gateway_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"""
  ForgeCode ↔ Databricks AI Gateway Bridge
  ─────────────────────────────────────────
  Listening on : http://{args.host}:{args.port}
  Gateway base : {DATABRICKS_BASE_URL}
  Models       : {len(MODELS)} available (from {_MODELS_SOURCE})
  Endpoints    : cursor -> {ENDPOINT_PATHS.get("cursor", _DEFAULT_ENDPOINT_PATHS["cursor"])}
                 openai -> {ENDPOINT_PATHS.get("openai", _DEFAULT_ENDPOINT_PATHS["openai"])}
                 mlflow -> {ENDPOINT_PATHS.get("mlflow", _DEFAULT_ENDPOINT_PATHS["mlflow"])}
  Timeouts     : upstream={UPSTREAM_TIMEOUT_SECONDS}s, max consecutive read timeouts={MAX_READ_TIMEOUTS}

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
