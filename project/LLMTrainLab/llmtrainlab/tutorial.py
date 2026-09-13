"""Hands-on path used by `llmctl tutorial` and the README.

Commands in WALK are what you type. `run_tutorial()` is the same story
without a shell, so tests can lock the expected phases.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llmtrainlab.engine import ClusterEngine

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

WALK = """
# LLMTrainLab tutorial — paste one block at a time from the repo root of this lab.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

# 1. SKU × model × method → recipe (no cluster yet)
llmctl landscape gpus
llmctl landscape parallelism
llmctl landscape recipe --model llama-70b --method full --gpu H100
llmctl landscape recipe --model llama-70b --method full --gpu H200
llmctl landscape recipe --model llama-8b --method lora
llmctl landscape recipe --model deepseek-v3 --method rlhf

# 2. Frontier cluster (A100 / H100 / H200 / B200 / L40S)
llmctl cluster init --config examples/cluster-frontier.yaml
llmctl kubectl get nodes

# 3. LoRA 8B → one L40S, no NVLink, no RDMA
llmctl job submit examples/jobs/llama8-lora.yaml
llmctl tick --steps 8
llmctl job list

# 4. 70B full FT → TP on one H200 NVLink island
llmctl job submit examples/jobs/llama70-full.yaml
llmctl tick --steps 6
llmctl job list

# 5. 70B vLLM replica on leftover H200
llmctl job submit examples/jobs/llama70-vllm.yaml
llmctl tick --steps 4
llmctl kubectl get pods

# 6. MoE RLHF wants 8×H200 + RDMA; 6 H200 already used → gang keeps it Queued
llmctl job submit examples/jobs/deepseek-rlhf.yaml
llmctl tick --steps 3
llmctl job list

# 7. TP>1 on PCIe L40S is a Filter miss, not a slow job
llmctl job submit examples/jobs/bad-tp-l40s.yaml
llmctl tick --steps 2
llmctl job list

# 8. Kill one LoRA rank after checkpoint; RestartAll the group
llmctl tick --steps 20
llmctl fault pod llama8-lora-worker-0
llmctl job status llama8-lora
llmctl tick --steps 8
llmctl job list

# 9. DCGM-shaped gauges + optional HTTP
llmctl metrics
llmctl kubectl get jobs
# pip install -e ".[api]" && llmctl serve
""".strip()


def run_tutorial() -> ClusterEngine:
    spec = yaml.safe_load((EXAMPLES / "cluster-frontier.yaml").read_text())
    engine = ClusterEngine(spec)

    def submit(name: str) -> None:
        engine.submit_job(yaml.safe_load((EXAMPLES / "jobs" / name).read_text()))

    submit("llama8-lora.yaml")
    engine.step(8)
    submit("llama70-full.yaml")
    engine.step(6)
    submit("llama70-vllm.yaml")
    engine.step(4)
    submit("deepseek-rlhf.yaml")
    engine.step(3)
    submit("bad-tp-l40s.yaml")
    engine.step(2)
    engine.step(20)
    engine.fault_pod("llama8-lora-worker-0")
    engine.step(8)
    return engine
