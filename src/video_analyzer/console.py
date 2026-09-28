"""Console output encoding on Windows.

A real console gets UTF-8 (Python writes through WriteConsoleW). A pipe (`| Out-Host`, `| Tee-Object`, `| more`)
is decoded by PowerShell/cmd with the console code page (often 850 or 1252): write in that one, with ASCII
fallbacks for symbols it lacks. ultralytics forces stdout to UTF-8 when imported on Windows
(ultralytics/utils/__init__.py, set_logging): call `setup_streams()` again after importing it.
"""
import codecs
import sys

_ASCII = {"▶": ">", "•": "-", "✔": "OK", "✗": "x", "⚠": "!", "—": "-", "–": "-", "…": "...", "→": "->",
          "·": "-", "’": "'", "«": '"', "»": '"'}


def _ascii_fallback(e: UnicodeEncodeError):
    return "".join(_ASCII.get(c, "?") for c in e.object[e.start:e.end]), e.end


def setup_streams() -> None:
    codecs.register_error("va_ascii", _ascii_fallback)
    piped = "utf-8"
    if sys.platform == "win32":
        import ctypes

        cp = ctypes.windll.kernel32.GetConsoleOutputCP()  # 0 when no console is attached
        if cp and cp != 65001:
            try:
                piped = codecs.lookup(f"cp{cp}").name
            except LookupError:
                pass
    for stream in (sys.stdout, sys.stderr):
        try:
            tty = stream.isatty()
            stream.reconfigure(encoding="utf-8" if tty else piped, errors="replace" if tty else "va_ascii")
        except (AttributeError, ValueError):
            pass
