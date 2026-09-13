from llmtrainlab.engine import ClusterEngine
from llmtrainlab.landscape import hbm_needed_gb, recommend
from llmtrainlab.tutorial import run_tutorial


def test_recipe_70b_full_uses_nvlink_tp():
    h100 = recommend("llama-70b", "full", "H100")
    h200 = recommend("llama-70b", "full", "H200")
    b200 = recommend("llama-70b", "full", "B200")
    assert h100.tp == 8
    assert h100.dp == 1
    assert h100.workers == 8
    assert h100.prefer_same_node
    assert hbm_needed_gb("llama-70b", "full", h100.tp) < 80
    assert h200.tp == 4
    assert h200.workers == 4
    assert b200.tp <= h200.tp


def test_tp_skips_pcie_nodes():
    engine = ClusterEngine(
        {
            "nodes": [
                {"name": "pcie", "gpu_type": "L40S", "gpus": 8, "network": "ethernet", "interconnect": "pcie"},
            ],
            "teams": {"team-a": {"namespace": "team-a", "guaranteed_gpus": 8, "burst_gpus": 8, "priority": 50}},
        }
    )
    engine.submit_job(
        {
            "metadata": {"name": "tp2", "namespace": "team-a"},
            "spec": {
                "team": "team-a",
                "gpuType": "L40S",
                "workers": 2,
                "gpusPerWorker": 1,
                "parallelism": {"tp": 2, "dp": 1, "pp": 1, "ep": 1},
                "steps": 10,
                "checkpointEvery": 5,
            },
        }
    )
    engine.step(4)
    assert engine.jobs["tp2"].phase == "Queued"
    assert engine.gpu_inventory()["L40S"]["used"] == 0


def test_ep_requires_rdma():
    engine = ClusterEngine(
        {
            "nodes": [
                {"name": "eth", "gpu_type": "H100", "gpus": 8, "network": "ethernet", "interconnect": "nvlink"},
            ],
            "teams": {"team-a": {"namespace": "team-a", "guaranteed_gpus": 8, "burst_gpus": 8, "priority": 50}},
        }
    )
    engine.submit_job(
        {
            "metadata": {"name": "moe", "namespace": "team-a"},
            "spec": {
                "team": "team-a",
                "model": "deepseek-v3",
                "method": "full",
                "gpuType": "H100",
                "steps": 10,
                "checkpointEvery": 5,
            },
        }
    )
    engine.step(4)
    job = engine.jobs["moe"]
    assert job.require_rdma
    assert job.ep >= 1
    assert job.phase == "Queued"


def test_h200_fits_70b_with_fewer_tp_than_h100():
    h100 = recommend("llama-70b", "infer", "H100")
    h200 = recommend("llama-70b", "infer", "H200")
    assert h200.tp <= h100.tp
    assert h200.hbm_gb_needed <= 141


def test_lora_recipe_lands_on_l40s():
    engine = ClusterEngine(
        {
            "nodes": [
                {"name": "inf", "gpu_type": "L40S", "gpus": 2, "network": "ethernet", "interconnect": "pcie"},
            ],
            "teams": {"team-a": {"namespace": "team-a", "guaranteed_gpus": 4, "burst_gpus": 4, "priority": 50}},
        }
    )
    engine.submit_job(
        {
            "metadata": {"name": "adapter", "namespace": "team-a"},
            "spec": {"team": "team-a", "model": "llama-8b", "method": "lora", "steps": 20, "checkpointEvery": 10},
        }
    )
    engine.step(5)
    assert engine.jobs["adapter"].phase in {"Starting", "Running"}
    assert engine.jobs["adapter"].gpu_type == "L40S"
    assert engine.jobs["adapter"].engine == "fsdp"


def test_tutorial_path_places_and_queues():
    engine = run_tutorial()
    lora = engine.jobs["llama8-lora"]
    full = engine.jobs["llama70-full"]
    infer = engine.jobs["llama70-vllm"]
    moe = engine.jobs["ds-rlhf"]
    bad = engine.jobs["bad-tp-l40s"]
    assert lora.gpu_type == "L40S"
    assert lora.retries >= 1
    assert full.gpu_type == "H200"
    assert full.tp == 4
    assert full.phase in {"Starting", "Running"}
    assert infer.engine == "vllm"
    assert infer.phase in {"Starting", "Running"}
    assert moe.require_rdma
    assert moe.ep >= 2
    assert moe.phase == "Queued"
    assert bad.phase == "Queued"
    assert engine.gpu_inventory()["L40S"]["used"] == 1
    assert lora.phase in {"Starting", "Running"}
