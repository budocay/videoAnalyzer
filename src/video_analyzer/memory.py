"""Memory helpers. MLX (mx.metal.* is deprecated in favour of mx.*); no-ops on the portable backend."""
import gc
import importlib.util

_HAS_MLX = importlib.util.find_spec("mlx") is not None


def free_mlx() -> None:
    gc.collect()
    if _HAS_MLX:
        import mlx.core as mx
        mx.clear_cache()


def peak_gb() -> float:
    if not _HAS_MLX:
        return float("nan")
    import mlx.core as mx
    return mx.get_peak_memory() / 1e9


def reset_peak() -> None:
    if _HAS_MLX:
        import mlx.core as mx
        mx.reset_peak_memory()
