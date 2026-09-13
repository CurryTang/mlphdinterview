"""Control-plane HTTP: cluster JSON + Prometheus text. Optional FastAPI extra."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def prometheus_text(metrics: dict[str, Any]) -> str:
    lines = [
        "# HELP llmtrainlab_gpu_slot_util allocated / allocatable GPU slots",
        "# TYPE llmtrainlab_gpu_slot_util gauge",
        f"llmtrainlab_gpu_slot_util {metrics.get('gpu_slot_util', 0)}",
        "# HELP llmtrainlab_queue_wait_p95 ticks",
        "# TYPE llmtrainlab_queue_wait_p95 gauge",
        f"llmtrainlab_queue_wait_p95 {metrics.get('queue_wait_p95', 0)}",
        "# HELP llmtrainlab_training_throughput steps per tick",
        "# TYPE llmtrainlab_training_throughput gauge",
        f"llmtrainlab_training_throughput {metrics.get('training_throughput', 0)}",
    ]
    for node, item in (metrics.get("dcgm") or {}).items():
        lines.append(
            f'llmtrainlab_dcgm_sm_util{{node="{node}",sku="{item["sku"]}"}} {item["sm_util"]}'
        )
        lines.append(
            f'llmtrainlab_dcgm_hbm_gb{{node="{node}",sku="{item["sku"]}"}} {item["hbm_gb"]}'
        )
    return "\n".join(lines) + "\n"


def run(host: str = "127.0.0.1", port: int = 8080, root: Path | None = None) -> None:
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, PlainTextResponse
        import uvicorn
    except ImportError as exc:
        raise SystemExit("pip install -e '.[api]'  to enable llmctl serve") from exc

    from llmtrainlab.store import load_state
    from llmtrainlab.engine import ClusterEngine

    app = FastAPI(title="LLMTrainLab control plane")
    dashboard = Path(__file__).with_name("dashboard.html")

    def _engine() -> ClusterEngine:
        return ClusterEngine.from_snapshot(load_state(root))

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return dashboard.read_text()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/cluster")
    def cluster() -> dict[str, Any]:
        return _engine().metrics()

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(prometheus_text(_engine().metrics()), media_type="text/plain; version=0.0.4")

    uvicorn.run(app, host=host, port=port)