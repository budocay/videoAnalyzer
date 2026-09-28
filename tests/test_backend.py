import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from video_analyzer import config as C
from video_analyzer.vision import OllamaVLM


class FakeOllama(BaseHTTPRequestHandler):
    requests = []

    def _reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # /api/tags
        self._reply({"models": [{"name": "qwen3-vl:4b-instruct", "model": "qwen3-vl:4b-instruct"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append((self.path, payload))
        if self.path == "/api/chat":
            self._reply({"message": {"content": "<think>x</think> Un carré rouge."}, "prompt_eval_count": 12,
                         "eval_count": 4, "done_reason": "stop"})
        else:
            self._reply({})

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeOllama.requests.clear()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_ollama_vlm_request_and_answer(server, tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"\xff\xd8fake")
    vlm = OllamaVLM("qwen3-vl:4b-instruct", url=server)
    a = vlm.ask("Décris.", [img], max_tokens=50)
    assert a.text == "Un carré rouge." and a.prompt_tokens == 12 and a.generation_tokens == 4
    path, p = FakeOllama.requests[-1]
    assert path == "/api/chat" and p["stream"] is False
    assert p["options"] == {"temperature": 0, "num_predict": 50}
    assert p["messages"][0]["images"] == [base64.b64encode(b"\xff\xd8fake").decode()]
    vlm.close()
    assert FakeOllama.requests[-1][1]["keep_alive"] == 0


def test_ollama_missing_model_is_explicit(server):
    with pytest.raises(RuntimeError, match="ollama pull qwen3-vl:8b-instruct"):
        OllamaVLM("qwen3-vl:8b-instruct", url=server)


def test_resolve_vlm_per_backend(monkeypatch):
    from video_analyzer import backend
    monkeypatch.setattr(C, "default_vlm_choice", lambda: None)  # ignore a local settings.json
    monkeypatch.setattr(backend, "name", lambda: "portable")
    monkeypatch.setattr(backend, "gpu_vram_gb", lambda: 12.0)
    assert C.resolve_vlm(None) == "qwen3-vl:8b-instruct"
    assert C.resolve_vlm("mlx-community/Qwen3.5-4B-MLX-8bit") == "qwen3-vl:4b-instruct"
    monkeypatch.setattr(backend, "gpu_vram_gb", lambda: 0.0)
    assert C.resolve_vlm(None) == "qwen3-vl:4b-instruct"
    monkeypatch.setattr(backend, "name", lambda: "mlx")
    assert C.resolve_vlm(None) == "mlx-community/Qwen3.5-9B-MLX-8bit"


def test_install_time_choice_is_the_default(monkeypatch):
    from video_analyzer import backend
    monkeypatch.setattr(backend, "name", lambda: "mlx")
    monkeypatch.setattr(C, "default_vlm_choice", lambda: "4b")
    assert C.resolve_vlm(None) == "mlx-community/Qwen3.5-4B-MLX-8bit"
    assert C.resolve_vlm("9b") == "mlx-community/Qwen3.5-9B-MLX-8bit"  # explicit --model wins


def test_amd_windows_vram_from_registry(monkeypatch):
    from video_analyzer import backend
    monkeypatch.setattr(backend, "_run", lambda cmd: "AMD Radeon RX 7900 XT|21458059264\r\n")
    g = backend._amd_windows()
    assert g.vendor == "amd" and g.name == "AMD Radeon RX 7900 XT" and round(g.vram_gb) == 20


def test_whisper_engine_per_gpu(monkeypatch):
    from video_analyzer import backend
    monkeypatch.delenv("VIDEO_ANALYZER_WHISPER", raising=False)
    monkeypatch.setattr(backend, "name", lambda: "portable")
    monkeypatch.setattr(backend, "gpu", lambda: backend.GPU("amd", "RX 7900 XT", 20.0))
    monkeypatch.setattr(backend, "torch_accel", lambda: "ROCm")
    assert backend.whisper_engine() == "transformers"
    monkeypatch.setattr(backend, "torch_accel", lambda: "CPU")   # ROCm PyTorch not working → CPU CT2
    assert backend.whisper_engine() == "faster-whisper"
    monkeypatch.setattr(backend, "gpu", lambda: backend.GPU("nvidia", "RTX 4070", 12.0))
    assert backend.whisper_engine() == "faster-whisper"


def test_amd_gpu_picks_the_big_model(monkeypatch):
    from video_analyzer import backend
    monkeypatch.setattr(C, "default_vlm_choice", lambda: None)
    monkeypatch.setattr(backend, "name", lambda: "portable")
    monkeypatch.setattr(backend, "gpu_vram_gb", lambda: 20.0)
    assert C.resolve_vlm(None) == "qwen3-vl:8b-instruct"
