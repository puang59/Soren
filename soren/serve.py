"""Serve the web visualizer with live analysis of a pasted function.

    python -m soren.serve                 # then open http://127.0.0.1:8000

The page in ``web-app/`` works as a static site on the saved test functions. Served from here
it also accepts a C or C++ function of your own: the function is parsed with Joern, turned
into a control flow graph, and walked by the trained agent and the declaring baselines. There
is no ground truth for pasted code, so a declaration is shown without a verdict.

The server listens on localhost only. It runs Joern on whatever is pasted, so do not expose
it to a network you do not trust.
"""

from __future__ import annotations

import argparse
import json
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from soren.data.live import MAX_CHARS, AnalysisError, analyse, find_joern
from soren.data.schema import GraphRecord
from soren.viz.webapp import Methods, graph_payload, load_methods

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 4 * MAX_CHARS
"""Request size limit in bytes; generous, since a character can take several bytes."""


class Analyser:
    """Parses a function and runs every method that needs no ground truth on it."""

    def __init__(self, methods: Methods, joern_home: Path | None) -> None:
        self.methods = methods
        self.joern_home = joern_home
        # One Joern at a time: each run starts a JVM, and the policy is not thread-safe.
        self._lock = threading.Lock()
        self._count = 0

    def status(self) -> dict[str, Any]:
        return {"live": self.joern_home is not None, "max_chars": MAX_CHARS}

    def parse(self, code: str) -> GraphRecord:
        if self.joern_home is None:
            raise AnalysisError("Joern is not installed, so pasted code cannot be analysed.")
        return analyse(code, self.joern_home)

    def episode(self, code: str) -> dict[str, Any]:
        with self._lock:
            graph = self.parse(code)
            self._count += 1
            payload = graph_payload(graph, self.methods.scorer)
            payload.update(id=f"pasted_{self._count}", project="your code", unlabelled=True)
            payload["traces"] = self.methods.traces(graph, labelled=False)
            return payload


class Handler(SimpleHTTPRequestHandler):
    analyser: Analyser

    def _json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/api/status":
            self._json(HTTPStatus.OK, self.analyser.status())
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/analyse":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            return self._json(
                HTTPStatus.BAD_REQUEST, {"error": "The request is empty or too large."}
            )
        try:
            code = json.loads(self.rfile.read(length))["code"]
            if not isinstance(code, str):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            return self._json(
                HTTPStatus.BAD_REQUEST, {"error": 'Send JSON of the form {"code": "..."}.'}
            )
        try:
            self._json(HTTPStatus.OK, self.analyser.episode(code))
        except AnalysisError as error:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})


def make_server(analyser: Analyser, host: str, port: int, directory: Path) -> ThreadingHTTPServer:
    handler = type("SorenHandler", (Handler,), {"analyser": analyser})
    return ThreadingHTTPServer((host, port), partial(handler, directory=str(directory)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default="runs/final/base/seed1/best_model.zip")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml", help="Protocol B thresholds")
    parser.add_argument("--joern-home", help="directory holding joern and joern-parse")
    parser.add_argument("--web-app", default=str(ROOT / "web-app"))
    args = parser.parse_args(argv)

    methods = load_methods(args.checkpoint, args.env_config, args.eval_config)
    if "ppo" not in methods.searchers:
        print(f"no checkpoint at {args.checkpoint}; pasted code will be walked by baselines only")
    joern_home = find_joern(args.joern_home)
    if joern_home is None:
        print("joern not found (set JOERN_HOME or pass --joern-home); live analysis is off")
    server = make_server(Analyser(methods, joern_home), args.host, args.port, Path(args.web_app))
    print(f"Soren is at http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
