"""VLM wrappers: load once, ask(prompt, images) many times. MLX (Apple Silicon) or Ollama (elsewhere).

Checked against mlx-vlm 0.7.2:
- load(repo) -> (model, processor); config is model.config.
- apply_chat_template(processor, config, prompt, num_images=N, **kw) inserts N image
  tokens (qwen3_5 = LIST_WITH_IMAGE_FIRST) and forwards kw to the HF chat template.
  It already defaults enable_thinking=False when the template accepts it; we pass it
  explicitly anyway and still strip <think> blocks as a safety net.
- generate(model, processor, prompt, image=[paths], max_tokens=, temperature=) -> GenerationResult
  with .text, .prompt_tokens, .generation_tokens, .generation_tps, .peak_memory (GB).
"""
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .memory import free_mlx

_THINK = re.compile(r"<think>.*?</think>", flags=re.S)


def clean(text: str) -> str:
    text = _THINK.sub("", text)
    text = re.sub(r"</?think>", "", text)
    return text.strip()


@dataclass(frozen=True)
class Answer:
    text: str
    seconds: float
    prompt_tokens: int
    generation_tokens: int
    peak_gb: float
    finish_reason: str | None


class VLM:
    def __init__(self, repo: str):
        from mlx_vlm import load

        self.repo = repo
        t = time.perf_counter()
        self.model, self.processor = load(repo)
        self.load_seconds = time.perf_counter() - t

    def ask(self, prompt: str, images: list[Path] | None = None, max_tokens: int = 400) -> Answer:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        paths = [str(p) for p in images or []]
        formatted = apply_chat_template(self.processor, self.model.config, prompt,
                                        num_images=len(paths), enable_thinking=False)
        t = time.perf_counter()
        out = generate(self.model, self.processor, formatted, image=paths or None,
                       max_tokens=max_tokens, temperature=0.0, verbose=False)
        return Answer(
            text=clean(out.text),
            seconds=time.perf_counter() - t,
            prompt_tokens=out.prompt_tokens,
            generation_tokens=out.generation_tokens,
            peak_gb=out.peak_memory,
            finish_reason=out.finish_reason,
        )

    def close(self) -> None:
        del self.model, self.processor
        free_mlx()


class OllamaVLM:
    """Same interface as VLM, served by a local Ollama (Windows / Linux / Intel Mac).

    POST /api/chat with base64 images; temperature 0 like the MLX path. Instruct tags of qwen3-vl
    have no thinking mode; <think> blocks are stripped anyway.
    """

    def __init__(self, model: str, url: str | None = None, timeout: float = 600):
        from . import backend

        self.repo = model
        self.url = (url or backend.OLLAMA_URL).rstrip("/")
        self.timeout = timeout
        t = time.perf_counter()
        tags = self._call("/api/tags", None, method="GET")
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        if model not in names:
            raise RuntimeError(f"modèle Ollama absent : lance `ollama pull {model}`")
        self._call("/api/generate", {"model": model, "prompt": "", "keep_alive": "30m"})  # load into memory
        self.load_seconds = time.perf_counter() - t

    def _call(self, path: str, payload: dict | None, method: str = "POST") -> dict:
        import json
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama injoignable sur {self.url} ({e}). Lance l'application Ollama "
                               "ou `ollama serve`.") from e
        except TimeoutError as e:  # connected, but no answer in time (e.g. cold model load)
            raise RuntimeError(f"Ollama n'a pas répondu en {self.timeout:.0f} s sur {path}") from e

    def ask(self, prompt: str, images: list[Path] | None = None, max_tokens: int = 400) -> Answer:
        import base64

        msg = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
        t = time.perf_counter()
        r = self._call("/api/chat", {"model": self.repo, "messages": [msg], "stream": False,
                                     "options": {"temperature": 0, "num_predict": max_tokens}})
        return Answer(
            text=clean(r.get("message", {}).get("content", "")),
            seconds=time.perf_counter() - t,
            prompt_tokens=int(r.get("prompt_eval_count", 0)),
            generation_tokens=int(r.get("eval_count", 0)),
            peak_gb=float("nan"),
            finish_reason=r.get("done_reason"),
        )

    def close(self) -> None:
        try:
            self._call("/api/generate", {"model": self.repo, "prompt": "", "keep_alive": 0})  # unload
        except RuntimeError:
            pass


def load_vlm(name: str):
    """MLX repo on Apple Silicon, Ollama tag elsewhere (see config.resolve_vlm)."""
    from . import backend

    return VLM(name) if backend.name() == "mlx" else OllamaVLM(name)
