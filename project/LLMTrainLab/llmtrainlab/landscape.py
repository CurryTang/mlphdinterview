"""AI & GPU landscape: SKUs, model families, finetune/infer recipes, fabric cost.

Numbers are order-of-magnitude planning values used by the simulator, not a
vendor datasheet. LLM weights never load; a recipe only changes placement,
gang size, and how many ticks a fake step costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

GPU_TYPES = ("A100", "H100", "H200", "B200", "L40S")


@dataclass(frozen=True)
class GpuSku:
    name: str
    hbm_gb: int
    hbm_tbs: float
    tdp_w: int
    nvlink: str
    host_attach: str
    nic: str
    notes: str
    good_for: tuple[str, ...]


GPUS: dict[str, GpuSku] = {
    "A100": GpuSku(
        "A100",
        80,
        2.0,
        400,
        "NVLink3 ~600 GB/s",
        "SXM / PCIe",
        "HDR 200G typical",
        "Still common. HBM is the limiter on 70B+ TP.",
        ("full-ft-small", "lora", "inference"),
    ),
    "H100": GpuSku(
        "H100",
        80,
        3.35,
        700,
        "NVLink4 ~900 GB/s",
        "SXM",
        "NDR 400G typical",
        "Transformer Engine / FP8. Same 80 GB as A100; bandwidth and NVLink are the upgrade.",
        ("full-ft", "rlhf-policy", "vllm"),
    ),
    "H200": GpuSku(
        "H200",
        141,
        4.8,
        700,
        "NVLink4 ~900 GB/s",
        "SXM (H100 HGX drop-in)",
        "NDR 400G typical",
        "Same NVLink generation as H100, much more HBM. Long context and MoE expert tables fit where H100 spills.",
        ("long-context-infer", "moe", "70b-full-fewer-tp"),
    ),
    "B200": GpuSku(
        "B200",
        192,
        8.0,
        1000,
        "NVLink5",
        "SXM (Blackwell)",
        "XDR / 800G class",
        "FP4/FP8 path. Fewer GPUs for the same model; NCCL + NVLink topology still dominate TP.",
        ("frontier-train", "large-moe", "fp4-infer"),
    ),
    "L40S": GpuSku(
        "L40S",
        48,
        0.86,
        350,
        "none (PCIe)",
        "PCIe",
        "host NIC only",
        "No NVLink domain. Fine for LoRA / vLLM on 8B–14B. TP>1 on this SKU is a product mistake.",
        ("lora", "qlora", "small-infer"),
    ),
}


@dataclass(frozen=True)
class ParallelAxis:
    name: str
    collective: str
    fabric: str
    scheduler: str


AXES: dict[str, ParallelAxis] = {
    "dp": ParallelAxis("data", "AllReduce grads", "ethernet acceptable if small; IB if many nodes", "replicas can span racks"),
    "tp": ParallelAxis("tensor", "AllGather / ReduceScatter", "NVLink domain; crossing NVLink kills MFU", "pack onto one node / NVLink island"),
    "pp": ParallelAxis("pipeline", "P2P activation send", "lower bandwidth than TP; bubble is the cost", "stages may span nodes"),
    "ep": ParallelAxis("expert", "AllToAll tokens", "RDMA / IB; ethernet AllToAll collapses", "Filter: requireRdma"),
}


@dataclass(frozen=True)
class Method:
    name: str
    workload: str
    hbm_mult_vs_infer: float
    gang: bool
    notes: str


METHODS: dict[str, Method] = {
    "full": Method("full", "training", 16.0, True, "Adam states + grads. Needs gang. Checkpoint is the RPO."),
    "lora": Method("lora", "training", 2.2, True, "Frozen base + adapters. Still a Worker group if TP>1."),
    "qlora": Method("qlora", "training", 1.3, True, "4-bit base. Fits L40S/A100 for 7B–32B. Slow backward."),
    "rlhf": Method("rlhf", "post-training", 4.0, True, "Policy + ref (+ reward). Rollouts are a second pool; do not Deployment-restart one rank."),
    "rlaif": Method("rlaif", "post-training", 3.5, True, "Same topology as RLHF; reward model instead of human labels."),
    "infer": Method("infer", "inference", 1.0, False, "Replica, not gang. KEDA on queue depth. KV cache is the HBM tax."),
}


@dataclass(frozen=True)
class Engine:
    name: str
    kind: str
    notes: str


ENGINES: dict[str, Engine] = {
    "fsdp": Engine("fsdp", "training", "ZeRO-3 style shard. AllReduce-heavy. Fine when TP=1."),
    "megatron": Engine("megatron", "training", "TP/PP/EP native. Maps onto NVLink + IB."),
    "vllm": Engine("vllm", "inference", "PagedAttention, continuous batch. Scale replicas, not ranks."),
    "sglang": Engine("sglang", "inference", "Radix prefix cache. Same replica model as vLLM for scheduling."),
    "trtllm": Engine("trtllm", "inference", "Compiled engine per SKU. Rebuild when GPU type changes."),
}


@dataclass(frozen=True)
class ModelFamily:
    name: str
    params_b: float
    dense: bool
    context: str
    notes: str


MODELS: dict[str, ModelFamily] = {
    "llama-8b": ModelFamily("llama-8b", 8, True, "8k–128k", "Single GPU inference. LoRA on L40S."),
    "llama-70b": ModelFamily("llama-70b", 70, True, "8k–128k", "TP=8 on H100 80GB; TP=4 on H200 141GB."),
    "qwen-32b": ModelFamily("qwen-32b", 32, True, "32k+", "TP=4 H100 or TP=2 H200."),
    "deepseek-v3": ModelFamily("deepseek-v3", 671, False, "128k", "MoE. EP AllToAll wants RDMA. Do not schedule as dense TP-only."),
}


@dataclass
class Recipe:
    model: str
    method: str
    engine: str
    gpu_type: str
    dp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    workers: int = 1
    gpus_per_worker: int = 1
    require_rdma: bool = False
    prefer_same_rack: bool = True
    prefer_same_node: bool = False
    qos: str = "training"
    hbm_gb_needed: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.ep


def effective_params_b(model: str) -> float:
    """MoE listed size is total experts; HBM tracks roughly-active params."""
    fam = MODELS[model]
    return fam.params_b if fam.dense else round(fam.params_b / 16.0, 1)


def hbm_needed_gb(model: str, method: str, tp: int, context_k: float = 8.0) -> float:
    bytes_per = {
        "infer": 2.4,
        "lora": 3.0,
        "qlora": 2.0,
        "full": 5.0,
        "rlhf": 10.0,
        "rlaif": 9.0,
    }[method]
    params = effective_params_b(model)
    kv = params * 0.04 * (context_k / 8.0) if method == "infer" else 0.0
    shard = max(1, tp)
    return round((params * bytes_per + kv) / shard, 1)


def min_tp(model: str, method: str, gpu_type: str) -> int:
    sku = GPUS[gpu_type]
    cap = sku.hbm_gb * 0.92
    nvlink = not sku.nvlink.startswith("none")
    for tp in (1, 2, 4, 8):
        if hbm_needed_gb(model, method, tp) > cap:
            continue
        if tp > 1 and not nvlink:
            continue
        return tp
    return 8


def recommend(model: str, method: str, gpu_type: str | None = None) -> Recipe:
    fam = MODELS[model]
    if method == "infer":
        engine = "vllm"
        qos = "inference"
        if fam.params_b <= 14 and fam.dense:
            gpu = gpu_type or "L40S"
        elif fam.params_b <= 40 and fam.dense:
            gpu = gpu_type or "H100"
        else:
            gpu = gpu_type or "H200"
        tp = min_tp(model, method, gpu)
        if tp > 1 and GPUS[gpu].nvlink.startswith("none"):
            gpu = gpu_type or "H100"
            tp = min_tp(model, method, gpu)
        if not fam.dense:
            gpu = gpu_type or "H200"
            tp = max(tp, 4)
        recipe = Recipe(model, method, engine, gpu, tp=tp, qos=qos, prefer_same_node=tp > 1)
        recipe.reasons.append("inference is replica-scoped; KEDA scales copies, kube-scheduler must not restart one TP rank")
    elif method in {"lora", "qlora"} and fam.params_b <= 14:
        gpu = gpu_type or "L40S"
        engine = "fsdp"
        recipe = Recipe(model, method, engine, gpu, qos="training")
        recipe.reasons.append("adapter FT fits one PCIe GPU; NVLink unused")
    elif not fam.dense:
        gpu = gpu_type or "H200"
        engine = "megatron"
        tp = max(4, min_tp(model, method, gpu))
        ep = 2
        recipe = Recipe(
            model,
            method,
            engine,
            gpu,
            dp=1,
            tp=tp,
            ep=ep,
            require_rdma=True,
            prefer_same_node=True,
            qos="post-training" if method in {"rlhf", "rlaif"} else "training",
        )
        recipe.reasons.append("MoE EP AllToAll → RDMA Filter")
        recipe.reasons.append(f"TP={tp} stays inside one NVLink island; EP={ep} needs IB")
        recipe.reasons.append("lab pack: one HGX (≤8 GPUs). Production adds DP islands.")
    else:
        gpu = gpu_type or ("H200" if fam.params_b >= 70 else "H100")
        sku = GPUS[gpu]
        tp = min_tp(model, method, gpu)
        engine = "megatron" if tp > 1 else "fsdp"
        recipe = Recipe(
            model,
            method,
            engine,
            gpu,
            dp=1,
            tp=tp,
            require_rdma=False,
            prefer_same_node=tp > 1,
            qos="post-training" if method in {"rlhf", "rlaif"} else "training",
        )
        recipe.reasons.append(f"TP={tp} so ~{hbm_needed_gb(model, method, tp)} GB/rank fits {sku.hbm_gb} GB {gpu}")
        recipe.reasons.append("pack one NVLink island (≤8 GPUs). Extra DP is a second island + RDMA.")
    recipe.workers = recipe.world_size
    recipe.hbm_gb_needed = hbm_needed_gb(model, method, max(1, recipe.tp))
    sku = GPUS[recipe.gpu_type]
    if recipe.hbm_gb_needed > sku.hbm_gb * 0.92:
        recipe.reasons.append(f"HBM tight: need ~{recipe.hbm_gb_needed} GB/rank on {sku.hbm_gb} GB {sku.name}")
    if recipe.tp > 1 and sku.nvlink.startswith("none"):
        recipe.reasons.append("TP>1 on a non-NVLink SKU will run; MFU will not")
    return recipe


def apply_recipe(spec: dict[str, Any]) -> dict[str, Any]:
    """Fill workers / gpu / fabric flags from model+method if present."""
    model = spec.get("model")
    method = spec.get("method")
    if not model or not method:
        return spec
    if model not in MODELS:
        raise ValueError(f"unknown model {model}; choose {sorted(MODELS)}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method}; choose {sorted(METHODS)}")
    gpu = spec.get("gpuType") or spec.get("gpu_type")
    recipe = recommend(model, method, gpu)
    spec.setdefault("gpuType", recipe.gpu_type)
    spec.setdefault("engine", recipe.engine)
    spec.setdefault("qos", recipe.qos)
    par = spec.setdefault("parallelism", {})
    par.setdefault("dp", recipe.dp)
    par.setdefault("tp", recipe.tp)
    par.setdefault("pp", recipe.pp)
    par.setdefault("ep", recipe.ep)
    world = int(par["dp"]) * int(par["tp"]) * int(par["pp"]) * int(par["ep"])
    spec.setdefault("workers", world)
    spec.setdefault("gpusPerWorker", 1)
    if recipe.require_rdma:
        spec["requireRdma"] = True
    if recipe.prefer_same_node:
        spec["preferSameNode"] = True
        spec.setdefault("preferSameRack", True)
    spec["_recipe_reasons"] = list(recipe.reasons)
    spec["_hbm_gb_needed"] = recipe.hbm_gb_needed
    return spec


def fabric_step_cost(job_tp: int, job_ep: int, job_dp: int, node_interconnects: Iterable[str], node_networks: Iterable[str], extra: int = 0) -> int:
    interconnects = set(node_interconnects)
    networks = set(node_networks)
    cost = extra
    if job_tp > 1 and ("nvlink" not in interconnects or len(interconnects) > 1):
        cost += 3
    if job_ep > 1 and "rdma" not in networks:
        cost += 4
    if job_dp > 1 and "ethernet" in networks and "rdma" not in networks:
        cost += 1
    return cost


def table_gpus() -> list[dict[str, Any]]:
    rows = []
    for sku in GPUS.values():
        rows.append(
            {
                "sku": sku.name,
                "hbm_gb": sku.hbm_gb,
                "hbm_tbs": sku.hbm_tbs,
                "nvlink": sku.nvlink,
                "nic": sku.nic,
                "tdp_w": sku.tdp_w,
                "notes": sku.notes,
            }
        )
    return rows


def table_axes() -> list[dict[str, str]]:
    return [
        {"axis": a.name, "collective": a.collective, "fabric": a.fabric, "scheduler": a.scheduler}
        for a in AXES.values()
    ]
