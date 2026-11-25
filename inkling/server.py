"""Loopback-only, single-request text chat server for Dari's Metal engine."""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import (
    load_tokenizer,
)  # Tokenization only; never loads the reference model.

from inkling.chat_protocol import InvalidGeneration, assistant_events, prepare
from inkling.cli import DEFAULT_CONTEXT_WINDOW, add_context_option
from inkling.decode_message_gate import GatedModel
from inkling.prefix_cache import PrefixCache

MODEL = "inkling-2-mini"


class Runtime:
    def __init__(self, snapshot, context_window=DEFAULT_CONTEXT_WINDOW):
        self.cache = PrefixCache()
        config = json.loads((snapshot / "config.json").read_text())
        maximum = config["text_config"]["model_max_length"]
        if not 1 <= context_window <= maximum:
            raise SystemExit(
                f"--context-window must be between 1 and {maximum:,} tokens"
            )
        self.model_max_length = maximum
        self.context_window = context_window
        self.softcap = config["text_config"]["final_logit_softcapping"]
        self.tokenizer = load_tokenizer(
            snapshot, eos_token_ids=[config["eos_token_id"]]
        )
        self.model = GatedModel(snapshot)

    def prompt(self, messages, tools):
        ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            return_dict=False,
            tokenize=True,
            add_generation_prompt=False,
            reasoning_effort="low" if tools else "none",
        )
        # With tools, let the model select text or a named invocation.
        return ids + self.tokenizer.encode(
            "<|message_model|>" + ("" if tools else "<|content_text|>"),
            add_special_tokens=False,
        )

    def generate(self, ids, limit, temperature):
        cursor = self.cache.resume(ids)
        self.cached_tokens = len(cursor.tokens)
        for offset in range(self.cached_tokens, len(ids), 1024):
            chunk = ids[offset : offset + 1024]
            cursor.logits = self.model(mx.array([chunk]), cursor.layers)
            mx.eval(cursor.logits)
            cursor.tokens.extend(chunk)
            if 0 < len(ids) - len(cursor.tokens) <= 1024:
                self.cache.block = cursor.checkpoint()
        self.cache.prompt = cursor.checkpoint()
        for _ in range(limit):
            scores = cursor.logits.reshape(-1)
            if temperature:
                scores = mx.tanh(scores.astype(mx.float32) / self.softcap) * (
                    self.softcap / temperature
                )
                token = mx.random.categorical(scores).item()
            else:
                token = mx.argmax(scores).item()
            # Only publish evaluated state. The yielded token itself is not
            # cached until generation resumes and the model consumes it.
            self.cache.tip = cursor
            yield token
            self.cache.tip = None
            cursor.logits = self.model(mx.array([[token]]), cursor.layers)
            cursor.tokens.append(token)


class ChatHandler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_HEAD(self):
        self.send_response(200 if self.path == "/" else 404)
        self.end_headers()

    def do_GET(self):
        if self.path != "/v1/models":
            return self.send_error(404)
        self.reply(
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": MODEL,
                        "object": "model",
                        "owned_by": "thinkingmachines",
                        "context_window": self.server.runtime.context_window,
                    }
                ],
            },
        )

    def do_POST(self):
        native = self.path in ("/api/chat", "/api/generate", "/api/show")
        if not native and self.path != "/v1/chat/completions":
            return self.send_error(404)
        if not native and self.headers.get_content_type() != "application/json":
            return self.send_error(415, "Expected application/json")
        if self.headers.get("Origin") or self.headers.get("Host", "").split(":")[
            0
        ] not in ("localhost", "127.0.0.1"):
            return self.send_error(403, "Local chat clients only")
        runtime = self.server.runtime
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4_000_000:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(size))
            if self.path == "/api/show":
                model = request.get("model") or request.get("name")
                if model not in (MODEL, MODEL + ":latest"):
                    return self.reply(404, {"error": f"model {model!r} not found"})
                return self.reply(
                    200,
                    {
                        "capabilities": ["completion"],
                        "details": {"family": "inkling", "format": "safetensors"},
                        "model_info": {
                            "general.architecture": "inkling",
                            "inkling.context_length": runtime.model_max_length,
                        },
                    },
                )
            if native:
                from inkling.ollama import generate

                return generate(self, request)
            if request["model"] != MODEL:
                raise ValueError(f"Expected model {MODEL}")
            messages, tools = prepare(request)
            stream_options = request.get("stream_options") or {}
            if not isinstance(stream_options, dict):
                raise TypeError("Expected stream_options object")
            limit = int(
                request.get("max_completion_tokens", request.get("max_tokens", 512))
            )
            temperature = float(request.get("temperature", 0))
            ids = runtime.prompt(messages, tools)
            if limit < 1 or not 0 <= temperature <= 2:
                raise ValueError(
                    "Expected positive output token budget and temperature between 0 and 2"
                )
            if len(ids) + limit > runtime.context_window:
                raise ValueError(
                    f"{len(ids)} prompt + {limit} requested output tokens exceed "
                    f"the {runtime.context_window:,}-token context window"
                )
        except (TypeError, ValueError, KeyError) as error:
            return self.reply(
                400,
                {
                    "error": str(error)
                    if native
                    else {"message": str(error), "type": "invalid_request_error"}
                },
            )

        meta = {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "model": MODEL,
            "created": int(time.time()),
        }
        output = assistant_events(
            runtime.tokenizer,
            runtime.generate(ids, limit, temperature),
            {t["function"]["name"] for t in tools},
            limit,
        )

        def usage(count):
            return {
                "prompt_tokens": len(ids),
                "prompt_tokens_details": {"cached_tokens": runtime.cached_tokens},
                "completion_tokens": count,
                "total_tokens": len(ids) + count,
            }

        if not request.get("stream"):
            parts = []
            calls = []
            try:
                for delta, reason, count in output:
                    if "content" in delta:
                        parts.append(delta["content"])
                    for call in delta.get("tool_calls", []):
                        calls.append({k: v for k, v in call.items() if k != "index"})
            except InvalidGeneration as error:
                return self.reply(
                    502,
                    {"error": {"message": str(error), "type": "invalid_model_output"}},
                )
            message = {
                "role": "assistant",
                "content": "".join(parts) if parts or not calls else None,
            }
            if calls:
                message["tool_calls"] = calls
            self.reply(
                200,
                dict(
                    **meta,
                    object="chat.completion",
                    choices=[{"index": 0, "message": message, "finish_reason": reason}],
                    usage=usage(count),
                ),
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(delta, finish=None):
            chunk = dict(
                **meta,
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": delta, "finish_reason": finish}],
            )
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())

        try:
            emit({"role": "assistant"})
            for delta, reason, count in output:
                if delta:
                    emit(delta)
                if reason:
                    emit({}, reason)
            if stream_options.get("include_usage"):
                chunk = dict(
                    **meta,
                    object="chat.completion.chunk",
                    choices=[],
                    usage=usage(count),
                )
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        except InvalidGeneration as error:
            body = json.dumps(
                {"error": {"message": str(error), "type": "invalid_model_output"}}
            )
            self.wfile.write(f"data: {body}\n\ndata: [DONE]\n\n".encode())
        except (BrokenPipeError, ConnectionResetError):
            pass  # The last completed prefix remains reusable.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8080)
    add_context_option(parser)
    args = parser.parse_args()
    if not args.snapshot.is_dir():
        parser.error("Expected an existing local snapshot directory")
    serve(args.snapshot, args.port, args.context_window)


def serve(snapshot, port, context_window):
    mx.set_cache_limit(512 * 1024**2)
    with HTTPServer(("127.0.0.1", port), ChatHandler) as server:
        server.runtime = Runtime(snapshot, context_window)
        print(
            f"Dari GatedModel ready on port {port}; context: {context_window:,} tokens; weights: {snapshot}",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
