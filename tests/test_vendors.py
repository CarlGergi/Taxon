"""Each client runs through its real SDK against a local stand-in for the vendor's API."""

import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from taxon.labeling.env import MissingKey, Secrets
from taxon.labeling.llm import CHOICE_SCHEMA, LLMRequest, ModelConfig, Status
from taxon.labeling.pipeline import ANNOTATORS, ARBITRATOR
from taxon.labeling.vendors import (
    ANTHROPIC_URL,
    GOOGLE_URL,
    OPENAI_URL,
    AnthropicBatchClient,
    GoogleBatchClient,
    OpenAIBatchClient,
    build_clients,
)

HAIKU, LUNA, FLASH = ANNOTATORS
KEY = "test-key-should-not-leak"


def request(config, text="hello", image=None):
    return LLMRequest(config, "pick", "stable system", text, image, "sha" if image else None)


class Api(BaseHTTPRequestHandler):
    routes = None

    def log_message(self, *args):
        pass

    def _send(self, status, payload, headers=None, raw=False):
        body = payload if raw else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _handle(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        headers = {name.lower(): value for name, value in self.headers.items()}
        self.server.calls.append((method, self.path, headers, body))
        for pattern, verb, handler in self.server.routes:
            if verb == method and re.fullmatch(pattern, self.path.split("?")[0]):
                return handler(self, body)
        self._send(404, {"error": {"message": f"no route {method} {self.path}"}})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


@pytest.fixture
def server():
    instance = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    instance.calls = []
    instance.routes = []
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    instance.url = f"http://127.0.0.1:{instance.server_port}"
    yield instance
    instance.shutdown()
    instance.server_close()


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "p.jpg"
    Image.new("RGB", (8, 8)).save(path, format="JPEG")
    return str(path)


# Anthropic


def anthropic_routes(server, rows, status="ended"):
    def batch(url):
        return {
            "id": "msgbatch_1",
            "type": "message_batch",
            "processing_status": status,
            "request_counts": {
                "processing": 0,
                "succeeded": len(rows),
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "created_at": "2026-11-03T00:00:00Z",
            "expires_at": "2026-11-04T00:00:00Z",
            "ended_at": "2026-11-03T00:10:00Z",
            "cancel_initiated_at": None,
            "archived_at": None,
            "results_url": url,
        }

    results_url = f"{server.url}/v1/messages/batches/msgbatch_1/results"
    jsonl = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    server.routes = [
        (r"/v1/messages/batches", "POST", lambda h, _: h._send(200, batch(None))),
        (r"/v1/messages/batches/msgbatch_1", "GET", lambda h, _: h._send(200, batch(results_url))),
        (
            r"/v1/messages/batches/msgbatch_1/results",
            "GET",
            lambda h, _: h._send(200, jsonl, raw=True),
        ),
    ]


def anthropic_row(key, text='{"choice": 2}', stop="end_turn", **usage):
    return {
        "custom_id": key,
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "stop_reason": stop,
                "stop_sequence": None,
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "s"},
                    {"type": "text", "text": text},
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 7,
                    **usage,
                },
            },
        },
    }


def test_anthropic_sends_the_cached_prefix_schema_effort_and_image(server, image):
    r = request(HAIKU, image=image)
    anthropic_routes(server, [anthropic_row(r.key)])
    client = AnthropicBatchClient(KEY, base_url=server.url)
    assert client.submit([r]) == "msgbatch_1"
    _, path, headers, body = server.calls[0]
    assert path == "/v1/messages/batches"
    assert headers["authorization"] == f"Bearer {KEY}"
    sent = json.loads(body)["requests"][0]
    assert sent["custom_id"] == r.key
    params = sent["params"]
    assert params["model"] == "claude-haiku-5-5"
    assert params["system"][0]["text"] == "stable system"
    assert params["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["schema"] == CHOICE_SCHEMA
    assert params["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"
    assert params["messages"][0]["content"][-1]["text"] == "hello"
    assert "temperature" not in params and "thinking" not in params


def test_anthropic_reads_results_past_thinking_blocks_with_cache_usage(server):
    r = request(HAIKU)
    anthropic_routes(server, [anthropic_row(r.key)])
    client = AnthropicBatchClient(KEY, base_url=server.url)
    assert client.poll("msgbatch_1") is True
    [result] = client.results("msgbatch_1")
    assert (result.key, result.status, result.text) == (r.key, Status.OK, '{"choice": 2}')
    assert result.usage.input_tokens == 10
    assert result.usage.cached_tokens == 100
    assert result.usage.cache_write_tokens == 7
    assert result.usage.output_tokens == 5


def test_anthropic_poll_is_false_while_running(server):
    anthropic_routes(server, [], status="in_progress")
    assert AnthropicBatchClient(KEY, base_url=server.url).poll("msgbatch_1") is False


@pytest.mark.parametrize(
    ("stop", "status"), [("refusal", Status.REFUSED), ("max_tokens", Status.TRUNCATED)]
)
def test_anthropic_maps_stop_reasons(server, stop, status):
    r = request(HAIKU)
    anthropic_routes(server, [anthropic_row(r.key, stop=stop)])
    [result] = AnthropicBatchClient(KEY, base_url=server.url).results("msgbatch_1")
    assert result.status is status


@pytest.mark.parametrize("kind", ["errored", "expired", "canceled"])
def test_anthropic_treats_other_outcomes_as_failed(server, kind):
    outcome = {"type": kind}
    if kind == "errored":
        outcome["error"] = {
            "type": "error",
            "error": {"type": "api_error", "message": "x"},
            "request_id": None,
        }
    anthropic_routes(server, [{"custom_id": "k", "result": outcome}])
    [result] = AnthropicBatchClient(KEY, base_url=server.url).results("msgbatch_1")
    assert result.status is Status.FAILED


def test_the_arbitrator_request_uses_sonnet_with_the_same_shape():
    params = AnthropicBatchClient.params(request(ARBITRATOR))
    assert params["model"] == "claude-sonnet-5-5"
    assert params["output_config"]["effort"] == "low"
    assert params["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "fallbacks" not in params
    assert "thinking" not in params


# OpenAI


def openai_routes(server, output_rows, error_rows=(), status="completed"):
    def batch(_h=None, _b=None):
        return {
            "id": "batch_1",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "file_in",
            "completion_window": "24h",
            "status": status,
            "created_at": 1,
            "output_file_id": "file_out" if output_rows else None,
            "error_file_id": "file_err" if error_rows else None,
        }

    def jsonl(rows):
        return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()

    server.routes = [
        (
            r"/v1/files",
            "POST",
            lambda h, _: h._send(
                200,
                {
                    "id": "file_in",
                    "object": "file",
                    "bytes": 1,
                    "created_at": 1,
                    "filename": "batch.jsonl",
                    "purpose": "batch",
                },
            ),
        ),
        (r"/v1/batches", "POST", lambda h, _: h._send(200, batch())),
        (r"/v1/batches/batch_1", "GET", lambda h, _: h._send(200, batch())),
        (
            r"/v1/files/file_out/content",
            "GET",
            lambda h, _: h._send(200, jsonl(output_rows), raw=True),
        ),
        (
            r"/v1/files/file_err/content",
            "GET",
            lambda h, _: h._send(200, jsonl(error_rows), raw=True),
        ),
    ]


def openai_row(key, content='{"choice": 3}', finish="stop", refusal=None):
    return {
        "id": "r",
        "custom_id": key,
        "error": None,
        "response": {
            "status_code": 200,
            "request_id": "req",
            "body": {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {"role": "assistant", "content": content, "refusal": refusal},
                    }
                ],
                "usage": {
                    "prompt_tokens": 110,
                    "completion_tokens": 6,
                    "total_tokens": 116,
                    "prompt_tokens_details": {"cached_tokens": 100},
                },
            },
        },
    }


def test_openai_uploads_one_line_per_request_with_schema_and_low_detail_image(server, image):
    r = request(LUNA, image=image)
    openai_routes(server, [openai_row(r.key)])
    client = OpenAIBatchClient(KEY, base_url=f"{server.url}/v1")
    assert client.submit([r]) == "batch_1"
    _, path, headers, upload = server.calls[0]
    assert path == "/v1/files"
    assert headers["authorization"] == f"Bearer {KEY}"
    line = next(x for x in upload.decode().splitlines() if x.startswith('{"custom_id"'))
    row = json.loads(line)
    assert row["custom_id"] == r.key
    assert row["url"] == "/v1/chat/completions"
    body = row["body"]
    assert body["model"] == "gpt-6-luna"
    assert body["messages"][0] == {"role": "system", "content": "stable system"}
    assert body["max_completion_tokens"] == 1024
    assert body["reasoning_effort"] == "low"
    assert body["response_format"]["json_schema"]["schema"] == CHOICE_SCHEMA
    assert body["messages"][1]["content"][1]["image_url"]["detail"] == "low"
    batch_call = json.loads(server.calls[1][3])
    assert batch_call["input_file_id"] == "file_in"
    assert batch_call["completion_window"] == "24h"


def test_openai_separates_cached_tokens_from_input(server):
    r = request(LUNA)
    openai_routes(server, [openai_row(r.key)])
    client = OpenAIBatchClient(KEY, base_url=f"{server.url}/v1")
    assert client.poll("batch_1") is True
    [result] = client.results("batch_1")
    assert (result.status, result.text) == (Status.OK, '{"choice": 3}')
    assert result.usage.input_tokens == 10
    assert result.usage.cached_tokens == 100
    assert result.usage.output_tokens == 6


def test_openai_poll_is_false_while_running(server):
    openai_routes(server, [], status="in_progress")
    assert OpenAIBatchClient(KEY, base_url=f"{server.url}/v1").poll("batch_1") is False


@pytest.mark.parametrize(
    ("row_args", "status"),
    [
        ({"finish": "length"}, Status.TRUNCATED),
        ({"finish": "content_filter"}, Status.REFUSED),
        ({"refusal": "no", "content": None}, Status.REFUSED),
    ],
)
def test_openai_maps_finish_reasons(server, row_args, status):
    openai_routes(server, [openai_row("k", **row_args)])
    [result] = OpenAIBatchClient(KEY, base_url=f"{server.url}/v1").results("batch_1")
    assert result.status is status


def test_openai_reads_the_error_file_too(server):
    bad = {
        "id": "r",
        "custom_id": "k2",
        "response": None,
        "error": {"code": "batch_expired", "message": "late"},
    }
    openai_routes(server, [openai_row("k1")], [bad])
    results = OpenAIBatchClient(KEY, base_url=f"{server.url}/v1").results("batch_1")
    assert {r.key: r.status for r in results} == {"k1": Status.OK, "k2": Status.FAILED}
    assert results[1].error == "batch_expired"


# Google


def google_routes(server, rows, state="JOB_STATE_SUCCEEDED", cache_status=200):
    def job(_h=None, _b=None):
        return {
            "name": "batches/job1",
            "metadata": {
                "name": "batches/job1",
                "state": state,
                "output": {"responsesFile": "files/out1"},
            },
        }

    def upload(handler, _):
        if handler.headers.get("X-Goog-Upload-Command") == "start":
            url = f"{server.url}/upload-session"
            return handler._send(200, {}, headers={"X-Goog-Upload-Url": url})
        return None

    def session(handler, _):
        file = {"name": "files/in1", "mimeType": "jsonl", "state": "ACTIVE"}
        handler._send(200, {"file": file}, headers={"X-Goog-Upload-Status": "final"})

    def cache(handler, _):
        if cache_status != 200:
            return handler._send(cache_status, {"error": {"code": cache_status, "message": "no"}})
        return handler._send(200, {"name": "cachedContents/c1", "model": "models/x"})

    jsonl = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    server.routes = [
        (r"/v1beta/cachedContents", "POST", cache),
        (r"/upload/v1beta/files", "POST", upload),
        (r"/upload-session", "POST", session),
        (r"/v1beta/models/.*:batchGenerateContent", "POST", lambda h, _: h._send(200, job())),
        (r"/v1beta/batches/job1", "GET", lambda h, _: h._send(200, job())),
        (
            r"/(download/)?v1beta/files/out1:download",
            "GET",
            lambda h, _: h._send(200, jsonl, raw=True),
        ),
    ]


def google_row(key, text='{"choice": 1}', reason="STOP"):
    return {
        "key": key,
        "response": {
            "candidates": [
                {
                    "finishReason": reason,
                    "content": {
                        "parts": [{"text": "hmm", "thought": True}, {"text": text}],
                        "role": "model",
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 110,
                "candidatesTokenCount": 4,
                "thoughtsTokenCount": 3,
                "cachedContentTokenCount": 100,
            },
        },
    }


def uploaded_lines(server):
    body = next(call[3] for call in server.calls if call[1].startswith("/upload-session"))
    return [json.loads(x) for x in body.decode().splitlines() if x]


def test_google_uploads_jsonl_uses_the_cache_and_creates_a_job(server, image):
    r = request(FLASH, image=image)
    google_routes(server, [google_row(r.key)])
    charges = []
    client = GoogleBatchClient(KEY, base_url=server.url, charge=lambda *a: charges.append(a))
    assert client.submit([r]) == "batches/job1"
    [line] = uploaded_lines(server)
    assert line["key"] == r.key
    body = line["request"]
    assert body["cachedContent"] == "cachedContents/c1"
    assert "systemInstruction" not in body
    assert body["generationConfig"]["maxOutputTokens"] == 1024
    assert body["generationConfig"]["responseJsonSchema"] == CHOICE_SCHEMA
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert body["contents"][0]["parts"][0]["inlineData"]["mimeType"] == "image/jpeg"
    assert body["contents"][0]["parts"][-1]["text"] == "hello"
    created = next(c for c in server.calls if "batchGenerateContent" in c[1])
    assert "gemini-3.5-flash-lite" in created[1]
    assert charges[0][0] == "google"
    assert charges[0][1] > 0


def test_google_creates_one_cache_per_prefix(server):
    google_routes(server, [])
    client = GoogleBatchClient(KEY, base_url=server.url)
    client.submit([request(FLASH, "a")])
    client.submit([request(FLASH, "b")])
    assert sum("cachedContents" in c[1] for c in server.calls) == 1


def test_google_falls_back_to_an_inline_prefix_if_the_cache_is_refused(server):
    google_routes(server, [], cache_status=400)
    charges = []
    client = GoogleBatchClient(KEY, base_url=server.url, charge=lambda *a: charges.append(a))
    client.submit([request(FLASH)])
    [line] = uploaded_lines(server)
    assert line["request"]["systemInstruction"]["parts"][0]["text"] == "stable system"
    assert "cachedContent" not in line["request"]
    assert charges == []


def test_google_reads_results_counting_thoughts_as_output(server):
    r = request(FLASH)
    google_routes(server, [google_row(r.key)])
    client = GoogleBatchClient(KEY, base_url=server.url)
    assert client.poll("batches/job1") is True
    [result] = client.results("batches/job1")
    assert (result.status, result.text) == (Status.OK, '{"choice": 1}')
    assert result.usage.input_tokens == 10
    assert result.usage.cached_tokens == 100
    assert result.usage.output_tokens == 7


def test_google_poll_is_false_while_running(server):
    google_routes(server, [], state="JOB_STATE_RUNNING")
    assert GoogleBatchClient(KEY, base_url=server.url).poll("batches/job1") is False


@pytest.mark.parametrize(
    ("reason", "status"), [("MAX_TOKENS", Status.TRUNCATED), ("SAFETY", Status.REFUSED)]
)
def test_google_maps_finish_reasons(server, reason, status):
    google_routes(server, [google_row("k", reason=reason)])
    [result] = GoogleBatchClient(KEY, base_url=server.url).results("batches/job1")
    assert result.status is status


def test_google_treats_a_blocked_prompt_and_a_status_line_correctly(server):
    blocked = {"key": "a", "response": {"promptFeedback": {"blockReason": "SAFETY"}}}
    failed = {"key": "b", "status": {"code": 13, "message": "internal"}}
    google_routes(server, [blocked, failed])
    results = GoogleBatchClient(KEY, base_url=server.url).results("batches/job1")
    assert {r.key: r.status for r in results} == {"a": Status.REFUSED, "b": Status.FAILED}


def test_google_returns_nothing_for_a_failed_job_so_the_runner_retries(server):
    google_routes(server, [], state="JOB_STATE_FAILED")
    assert GoogleBatchClient(KEY, base_url=server.url).results("batches/job1") == []


# Shared


def test_a_shell_variable_cannot_redirect_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://attacker.example")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://attacker.example")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "http://attacker.example")
    assert "attacker" not in str(AnthropicBatchClient(KEY).client.base_url)
    assert "attacker" not in str(OpenAIBatchClient(KEY).client.base_url)
    assert ANTHROPIC_URL in str(AnthropicBatchClient(KEY).client.base_url)
    assert OPENAI_URL in str(OpenAIBatchClient(KEY).client.base_url)
    google = GoogleBatchClient(KEY)
    assert google.client._api_client.get_read_only_http_options()["base_url"].startswith(GOOGLE_URL)


def test_build_clients_names_the_missing_key():
    with pytest.raises(MissingKey, match="OPENAI_API_KEY"):
        build_clients(Secrets({"ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "b"}))


def test_build_clients_makes_all_three():
    clients = build_clients(
        Secrets({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "b", "GEMINI_API_KEY": "c"})
    )
    assert set(clients) == {"anthropic", "openai", "google"}


def test_keys_do_not_appear_in_request_bodies(server):
    r = request(HAIKU)
    anthropic_routes(server, [anthropic_row(r.key)])
    AnthropicBatchClient(KEY, base_url=server.url).submit([r])
    assert KEY.encode() not in server.calls[0][3]


def test_model_config_is_hashable_for_grouping():
    assert {ModelConfig("a", "b", 1, "low"): 1}


def test_a_jpeg_fixture_is_valid(image):
    with open(image, "rb") as f:
        assert Image.open(io.BytesIO(f.read())).format == "JPEG"
