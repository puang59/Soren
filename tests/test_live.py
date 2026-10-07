import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from soren.data.build import BuildConfig
from soren.data.live import MAX_CHARS, SAMPLE_ID, AnalysisError, build_record, normalise
from soren.data.schema import SchemaError
from soren.serve import Analyser, make_server
from soren.viz.webapp import load_methods

FIXTURES = Path(__file__).parent / "fixtures" / "joern"


def export_of(name: str, tmp_path: Path) -> tuple[str, Path]:
    """A fixture snippet and its Joern export, renamed as if it had just been pasted."""
    out = tmp_path / "out.jsonl"
    with open(FIXTURES / "snippets.jsonl", encoding="utf-8") as fh, open(out, "w") as sink:
        for line in fh:
            method = json.loads(line)
            path = Path(method["file"])
            if path.stem == name:
                method["file"] = f"{SAMPLE_ID}{path.suffix}"
                sink.write(json.dumps(method) + "\n")
    return (FIXTURES / "src" / f"{name}.c").read_text(), out


def test_normalise_rejects_empty_and_oversized_input():
    with pytest.raises(AnalysisError, match="Paste"):
        normalise("  \n\n")
    with pytest.raises(AnalysisError, match="too long"):
        normalise("x" * (MAX_CHARS + 1))
    assert normalise("\r\nint f() {\r\n}\r\n\r\n") == "int f() {\n}\n"


def test_build_record_gives_an_unlabelled_graph(tmp_path):
    code, export = export_of("if_else", tmp_path)
    record = build_record(code, export, BuildConfig(min_nodes=3))
    assert record.vuln_nodes == []
    assert record.nodes[record.entry].kind == "ENTRY"
    assert record.source_lines == code.splitlines()
    # Structurally sound, but not a labelled record.
    record.validate(labelled=False)
    with pytest.raises(SchemaError, match="vuln_nodes is empty"):
        record.validate()


def test_build_record_explains_what_went_wrong(tmp_path):
    code, export = export_of("if_else", tmp_path)
    with pytest.raises(AnalysisError, match="too short"):
        build_record(code, export, BuildConfig(min_nodes=500))
    with pytest.raises(AnalysisError, match="too long"):
        build_record(code, export, BuildConfig(min_nodes=3, max_nodes=3))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(AnalysisError, match="No function found"):
        build_record(code, empty)


@pytest.fixture(scope="module")
def methods():
    return load_methods("no-such-checkpoint.zip")


def test_unlabelled_traces_leave_out_the_oracle_stop(methods, tmp_path):
    code, export = export_of("if_else", tmp_path)
    record = build_record(code, export, BuildConfig(min_nodes=3))
    traces = methods.traces(record, labelled=False)
    assert traces and all(methods.info[name]["rule"] == "threshold" for name in traces)
    assert set(traces) < set(methods.searchers)


class StubAnalyser(Analyser):
    def __init__(self, methods, record):
        super().__init__(methods, joern_home=Path("."))
        self.record = record

    def parse(self, code):
        if "boom" in code:
            raise AnalysisError("Joern could not parse this code.")
        return self.record


@pytest.fixture
def server(methods, tmp_path):
    code, export = export_of("if_else", tmp_path)
    record = build_record(code, export, BuildConfig(min_nodes=3))
    (tmp_path / "index.html").write_text("<p>page</p>")
    httpd = make_server(StubAnalyser(methods, record), "127.0.0.1", 0, tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def post(url: str, body: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_server_serves_the_page_and_reports_status(server):
    with urllib.request.urlopen(f"{server}/index.html", timeout=10) as response:
        assert response.read() == b"<p>page</p>"
    with urllib.request.urlopen(f"{server}/api/status", timeout=10) as response:
        assert json.load(response) == {"live": True, "max_chars": MAX_CHARS}


def test_server_analyses_a_pasted_function(server):
    status, episode = post(f"{server}/api/analyse", json.dumps({"code": "int f() {}"}).encode())
    assert status == 200
    assert episode["unlabelled"] is True and episode["vulnerable"] == []
    assert episode["id"] == "pasted_1"
    assert len(episode["scores"]) == len(episode["nodes"])
    assert all(trace["steps"] for trace in episode["traces"].values())


def test_server_reports_errors_as_json(server):
    assert post(f"{server}/api/analyse", b"not json")[0] == 400
    assert post(f"{server}/api/analyse", json.dumps({"code": 3}).encode())[0] == 400
    assert post(f"{server}/api/nope", b"{}")[0] == 404
    status, body = post(f"{server}/api/analyse", json.dumps({"code": "boom"}).encode())
    assert status == 422 and body == {"error": "Joern could not parse this code."}
