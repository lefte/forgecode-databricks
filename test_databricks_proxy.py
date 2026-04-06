import unittest

from databricks_proxy import (
    _transform_upstream_chunk,
    adapt_payload_for_endpoint,
    convert_chat_to_responses_payload,
    extract_gateway_base_url,
    parse_models_config,
    resolve_endpoint_path,
    sanitize_payload,
)


class DatabricksProxyTests(unittest.TestCase):
    def test_parse_legacy_models_list_infers_endpoint_aliases(self):
        models, model_aliases, endpoint_paths = parse_models_config([
            "databricks-gpt-5-3-codex",
            "databricks-claude-haiku-4-5",
        ])

        self.assertEqual(models, ["databricks-gpt-5-3-codex", "databricks-claude-haiku-4-5"])
        self.assertEqual(model_aliases["databricks-gpt-5-3-codex"], "cursor")
        self.assertEqual(model_aliases["databricks-claude-haiku-4-5"], "mlflow")
        self.assertEqual(endpoint_paths["cursor"], "/cursor/v1/chat/completions")
        self.assertEqual(endpoint_paths["openai"], "/openai/v1/responses")

    def test_parse_structured_models_config_uses_endpoint_overrides(self):
        models, model_aliases, endpoint_paths = parse_models_config({
            "endpoints": {
                "openai": "openai/v1/responses",
                "custom": "/custom/v1/chat/completions",
            },
            "models": [
                {"id": "databricks-gpt-5-3-codex", "endpoint": "openai"},
                {"id": "my-special-model", "endpoint": "custom"},
            ],
        })

        self.assertEqual(models, ["databricks-gpt-5-3-codex", "my-special-model"])
        self.assertEqual(model_aliases["my-special-model"], "custom")
        self.assertEqual(endpoint_paths["openai"], "/openai/v1/responses")
        self.assertEqual(endpoint_paths["custom"], "/custom/v1/chat/completions")

    def test_resolve_endpoint_path_falls_back_to_inferred_alias(self):
        endpoint_path = resolve_endpoint_path(
            "databricks-gpt-5-1",
            {"databricks-gpt-5-1": "missing"},
            {"mlflow": "/mlflow/v1/chat/completions", "cursor": "/cursor/v1/chat/completions"},
        )
        self.assertEqual(endpoint_path, "/cursor/v1/chat/completions")

    def test_resolve_endpoint_path_uses_cursor_for_chat_payload(self):
        endpoint_path = resolve_endpoint_path(
            "databricks-gpt-5-3-codex",
            {"databricks-gpt-5-3-codex": "openai"},
            {
                "openai": "/openai/v1/responses",
                "cursor": "/cursor/v1/chat/completions",
            },
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(endpoint_path, "/cursor/v1/chat/completions")

    def test_resolve_endpoint_path_uses_openai_for_responses_payload(self):
        endpoint_path = resolve_endpoint_path(
            "databricks-gpt-5-3-codex",
            {"databricks-gpt-5-3-codex": "cursor"},
            {
                "openai": "/openai/v1/responses",
                "cursor": "/cursor/v1/chat/completions",
            },
            {"input": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(endpoint_path, "/openai/v1/responses")

    def test_extract_gateway_base_url_strips_path(self):
        base_url = extract_gateway_base_url(
            "https://123456789.ai-gateway.azuredatabricks.net/mlflow/v1/chat/completions"
        )
        self.assertEqual(base_url, "https://123456789.ai-gateway.azuredatabricks.net")

    def test_convert_chat_to_responses_payload(self):
        converted = convert_chat_to_responses_payload({
            "model": "databricks-gpt-5-3-codex",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        })

        self.assertNotIn("messages", converted)
        self.assertIn("input", converted)
        self.assertEqual(converted["max_output_tokens"], 128)
        self.assertEqual(
            converted["input"],
            [
                {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "hi there"}]},
            ],
        )

    def test_adapt_payload_for_openai_endpoint_converts_chat_payload(self):
        adapted = adapt_payload_for_endpoint(
            {
                "model": "databricks-gpt-5-3-codex",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
            "/openai/v1/responses",
            {
                "openai": "/openai/v1/responses",
                "cursor": "/cursor/v1/chat/completions",
            },
        )

        self.assertNotIn("messages", adapted)
        self.assertEqual(adapted["max_output_tokens"], 64)
        self.assertEqual(adapted["input"], [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}])

    def test_adapt_payload_for_cursor_endpoint_converts_chat_payload(self):
        adapted = adapt_payload_for_endpoint(
            {
                "model": "databricks-gpt-5-3-codex",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
            "/cursor/v1/chat/completions",
            {
                "openai": "/openai/v1/responses",
                "cursor": "/cursor/v1/chat/completions",
            },
        )

        self.assertNotIn("messages", adapted)
        self.assertEqual(adapted["max_output_tokens"], 64)
        self.assertEqual(adapted["input"], [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}])

    def test_adapt_payload_for_cursor_endpoint_normalizes_tools_shape(self):
        adapted = adapt_payload_for_endpoint(
            {
                "model": "databricks-gpt-5-3-codex",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "read_file"},
                },
            },
            "/cursor/v1/chat/completions",
            {
                "openai": "/openai/v1/responses",
                "cursor": "/cursor/v1/chat/completions",
            },
        )

        self.assertEqual(
            adapted["tools"],
            [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        self.assertEqual(adapted["tool_choice"], {"type": "function", "name": "read_file"})

    def test_transform_upstream_chunk_maps_responses_delta_to_chat_chunk(self):
        transformed, needs_role_chunk = _transform_upstream_chunk(
            raw_chunk={"type": "response.output_text.delta", "delta": "hello"},
            req_id="chatcmpl-test",
            created_time=123,
            req_model="databricks-gpt-5-3-codex",
            needs_role_chunk=True,
        )

        self.assertFalse(needs_role_chunk)
        self.assertEqual(len(transformed), 1)
        self.assertEqual(transformed[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(transformed[0]["choices"][0]["delta"]["content"], "hello")
        self.assertIsNone(transformed[0]["choices"][0]["finish_reason"])

    def test_transform_upstream_chunk_ignores_response_created_event(self):
        transformed, needs_role_chunk = _transform_upstream_chunk(
            raw_chunk={"type": "response.created", "response": {"id": "resp_123"}},
            req_id="chatcmpl-test",
            created_time=123,
            req_model="databricks-gpt-5-3-codex",
            needs_role_chunk=True,
        )

        self.assertEqual(transformed, [])
        self.assertTrue(needs_role_chunk)

    def test_sanitize_payload_removes_unsupported_keys(self):
        payload, model_id = sanitize_payload({
            "model": "databricks-gpt-5-3-codex",
            "stream": True,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 256,
            "parallel_tool_calls": True,
            "metadata": {"foo": "bar"},
        })

        self.assertEqual(model_id, "databricks-gpt-5-3-codex")
        self.assertEqual(payload["max_tokens"], 256)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("parallel_tool_calls", payload)
        self.assertNotIn("metadata", payload)
        self.assertNotIn("response_format", payload)


if __name__ == "__main__":
    unittest.main()
