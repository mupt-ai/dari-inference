"""Standalone launcher. Serving only reads local files; install is explicit."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

MODEL = "inkling-2-mini"
REPOSITORY = "thinkingmachines/Inkling-2-Mini"
REVISION = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
DEFAULT_CONTEXT_WINDOW = 50_000


def add_context_option(parser):
    parser.add_argument(
        "--context-window",
        type=int,
        default=os.environ.get("DARI_CONTEXT_LENGTH", DEFAULT_CONTEXT_WINDOW),
        metavar="TOKENS",
        help="Maximum combined prompt and output tokens (default: %(default)s)",
    )


def official_snapshot(install):
    def fetch(patterns):
        return Path(
            snapshot_download(
                REPOSITORY,
                revision=REVISION,
                allow_patterns=patterns,
                local_files_only=not install,
            )
        )

    # The engine only needs the text shards, not auxiliary model heads.
    metadata = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    root = fetch(metadata)
    for name in metadata:
        if not (root / name).is_file():
            raise FileNotFoundError(name)
    weights = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if "model.llm.embed.weight" not in weights:
        raise ValueError("Expected the official Inkling text checkpoint")
    shards = sorted({v for k, v in weights.items() if k.startswith("model.llm.")})
    if install:
        fetch(shards)
    for name in shards:
        if not (root / name).is_file():
            raise FileNotFoundError(name)
    return root


def main():
    parser = argparse.ArgumentParser(
        prog="dari-inference",
        description="Serve official Inkling weights offline on Dari Metal.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default=MODEL,
        choices=[MODEL, "install", "serve", "run", "pull", "list", "show"],
    )
    parser.add_argument("model", nargs="?", choices=[MODEL])
    parser.add_argument(
        "--snapshot", type=Path, help="Use an existing local checkpoint"
    )
    parser.add_argument("--port", type=int)
    add_context_option(parser)
    args = parser.parse_args()
    if args.action in ("run", "pull", "show") and not args.model:
        parser.error(f"{args.action} requires a model name")
    port = args.port if args.port is not None else 8081
    if args.action in ("list", "show"):
        try:
            snapshot = official_snapshot(False)
        except (LocalEntryNotFoundError, FileNotFoundError):
            if args.action == "list":
                print("NAME")
                return
            parser.exit(1, f"Model not installed: {MODEL}\n")
        if args.action == "list":
            print(f"NAME\n{MODEL}")
        else:
            maximum = json.loads((snapshot / "config.json").read_text())["text_config"][
                "model_max_length"
            ]
            print(
                f"Model: {MODEL}\nEngine: Dari Metal\nWeights: {REPOSITORY}@{REVISION}\nModel context limit: {maximum:,} tokens"
            )
        return
    install = args.action in ("install", "pull")
    if args.model and not install and args.action != "run":
        parser.error("Specify the model only once")
    if install and args.snapshot:
        parser.error("--snapshot is for serving existing weights")
    try:
        snapshot = args.snapshot or official_snapshot(install)
    except (LocalEntryNotFoundError, FileNotFoundError) as exc:
        if install:
            parser.exit(1, f"Installation incomplete: {exc}\n")
        parser.exit(
            1,
            "Official weights not installed. Run: dari-inference install (./dari-inference install in a checkout).\n",
        )
    if install:
        print(f"Ready: {MODEL} at {snapshot}")
        return
    from inkling.server import serve

    serve(snapshot, port, args.context_window)


if __name__ == "__main__":
    main()
