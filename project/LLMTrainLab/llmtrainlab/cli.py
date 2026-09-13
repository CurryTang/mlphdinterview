"""llmctl: walk the local LLM training control plane from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from llmtrainlab.engine import ClusterEngine, default_cluster_spec
from llmtrainlab.store import load_state, save_state, state_path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _load_engine(root: Path | None = None) -> ClusterEngine:
    return ClusterEngine.from_snapshot(load_state(root))


def _save(engine: ClusterEngine, root: Path | None = None) -> None:
    save_state(engine.snapshot(), root)


def _print_table(rows: list[dict[str, Any]], keys: list[str]) -> None:
    if not rows:
        print("(empty)")
        return
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows)) for key in keys}
    print("  ".join(key.ljust(widths[key]) for key in keys))
    print("  ".join("-" * widths[key] for key in keys))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key in keys))


def cmd_cluster_init(args: argparse.Namespace) -> int:
    spec = default_cluster_spec()
    if args.config:
        spec = yaml.safe_load(Path(args.config).read_text())
    engine = ClusterEngine(spec)
    _save(engine, args.root)
    inv = engine.gpu_inventory()
    print(f"cluster initialized → {state_path(args.root)}")
    print("GPU inventory:")
    for gpu_type, item in inv.items():
        print(f"  {gpu_type}: allocatable={item['allocatable']} total={item['total']}")
    print("nodes:", ", ".join(engine.nodes))
    print("teams:", ", ".join(engine.teams))
    return 0


def cmd_cluster_status(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    print(f"tick={engine.tick}")
    print("nodes:")
    _print_table(
        [
            {
                "name": node.name,
                "type": node.gpu_type,
                "gpus": node.gpus,
                "status": node.status,
                "rack": node.rack,
                "net": node.network,
                "kind": node.kind,
            }
            for node in engine.nodes.values()
        ],
        ["name", "type", "gpus", "status", "rack", "net", "kind"],
    )
    print("\njobs:")
    cmd_job_list(args)
    return 0


def cmd_job_submit(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    raw = yaml.safe_load(Path(args.file).read_text())
    job = engine.submit_job(raw)
    if not args.no_tick:
        engine.step(1)
    _save(engine, args.root)
    live = engine.jobs[job.name]
    print(f"submitted {live.name} phase={live.phase} demand={live.gpu_demand}x{live.gpu_type} {live.method} tp={live.tp} ep={live.ep}")
    for event in live.events:
        if event.startswith("t="):
            continue
        print(f"  why: {event}")
    return 0


def cmd_job_list(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    rows = []
    for job in engine.jobs.values():
        ckpt = engine._latest_ckpt(job.name)
        live = [w for w in engine.workers.values() if w.job == job.name and w.phase not in {"Stopped"}]
        step = max((w.step for w in live), default=(ckpt.step if ckpt else 0))
        rows.append(
            {
                "name": job.name,
                "ns": job.namespace,
                "phase": job.phase,
                "method": job.method,
                "engine": job.engine or "-",
                "gpus": job.gpu_demand,
                "type": job.gpu_type,
                "tp": job.tp,
                "step": step,
                "ckpt": ckpt.step if ckpt else "-",
                "retries": job.retries,
            }
        )
    _print_table(rows, ["name", "ns", "phase", "method", "engine", "gpus", "type", "tp", "step", "ckpt", "retries"])
    return 0


def cmd_job_status(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    job = engine.jobs[args.name]
    print(json.dumps({**engine.metrics()["jobs"][job.name], "events": job.events[-12:]}, indent=2))
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    engine.step(args.steps)
    _save(engine, args.root)
    print(f"tick={engine.tick}")
    cmd_job_list(args)
    return 0


def cmd_fault(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    kind = args.kind
    if kind == "pod":
        engine.fault_pod(args.target)
    elif kind == "node":
        engine.fault_node(args.target)
    elif kind == "network":
        if not args.job:
            raise SystemExit("fault network 需要 --job")
        engine.fault_network(args.job)
    elif kind == "ckpt":
        engine.fault_checkpoint_store(args.duration)
    elif kind == "gpu":
        engine.shrink_gpus(args.target, args.remaining)
    else:
        raise SystemExit(f"unknown fault {kind}")
    engine.step(args.steps)
    _save(engine, args.root)
    print(f"fault {kind} applied, tick={engine.tick}")
    cmd_job_list(args)
    return 0


def cmd_recover_node(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    engine.recover_node(args.name)
    engine.step(1)
    _save(engine, args.root)
    print(f"node {args.name} Ready, tick={engine.tick}")
    return 0


def cmd_scale(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    created = engine.scale_virtual_nodes(args.nodes, gpus=args.gpus, gpu_type=args.gpu_type)
    engine.step(1)
    _save(engine, args.root)
    print(f"added {created} KWOK-style virtual nodes, total nodes={len(engine.nodes)}")
    print(engine.gpu_inventory())
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    metrics = engine.metrics()
    if args.json:
        print(json.dumps(metrics, indent=2))
        return 0
    print(f"tick={metrics['tick']}  gpu_util={metrics['gpu_slot_util']}  pending_pods={metrics['pending_pods']}")
    print(f"queue wait   p50={metrics['queue_wait_p50']}  p95={metrics['queue_wait_p95']}")
    print(f"admit        p50={metrics['admit_latency_p50']}  p95={metrics['admit_latency_p95']}")
    print(f"recovery     p50={metrics['recovery_p50']}  p95={metrics['recovery_p95']}")
    print(f"throughput   {metrics['training_throughput']} steps/tick")
    print("tenant GPUs:", metrics["tenant_gpus"])
    print("inventory:", metrics["gpu_inventory"])
    if metrics.get("dcgm"):
        print("dcgm (HBM / NVLink):")
        _print_table(
            [
                {"node": name, **{k: item[k] for k in ("sku", "hbm_gb", "sm_util", "nvlink_active")}}
                for name, item in metrics["dcgm"].items()
            ],
            ["node", "sku", "hbm_gb", "sm_util", "nvlink_active"],
        )
    return 0


def cmd_landscape(args: argparse.Namespace) -> int:
    from llmtrainlab.landscape import (
        AXES,
        ENGINES,
        GPUS,
        METHODS,
        MODELS,
        recommend,
        table_axes,
        table_gpus,
    )

    topic = args.topic
    if topic == "gpus":
        _print_table(table_gpus(), ["sku", "hbm_gb", "hbm_tbs", "nvlink", "nic", "tdp_w"])
        for sku in GPUS.values():
            print(f"  {sku.name}: {sku.notes}")
        return 0
    if topic == "models":
        rows = [{"name": m.name, "params_b": m.params_b, "dense": m.dense, "context": m.context} for m in MODELS.values()]
        _print_table(rows, ["name", "params_b", "dense", "context"])
        return 0
    if topic == "methods":
        rows = [{"name": m.name, "workload": m.workload, "gang": m.gang, "notes": m.notes} for m in METHODS.values()]
        _print_table(rows, ["name", "workload", "gang", "notes"])
        return 0
    if topic == "engines":
        rows = [{"name": e.name, "kind": e.kind, "notes": e.notes} for e in ENGINES.values()]
        _print_table(rows, ["name", "kind", "notes"])
        return 0
    if topic == "parallelism":
        _print_table(table_axes(), ["axis", "collective", "fabric", "scheduler"])
        return 0
    if topic == "recipe":
        recipe = recommend(args.model, args.method, args.gpu)
        print(json.dumps(
            {
                "model": recipe.model,
                "method": recipe.method,
                "engine": recipe.engine,
                "gpuType": recipe.gpu_type,
                "parallelism": {"dp": recipe.dp, "tp": recipe.tp, "pp": recipe.pp, "ep": recipe.ep},
                "workers": recipe.workers,
                "requireRdma": recipe.require_rdma,
                "preferSameNode": recipe.prefer_same_node,
                "qos": recipe.qos,
                "hbm_gb_needed": recipe.hbm_gb_needed,
                "why": recipe.reasons,
            },
            indent=2,
        ))
        return 0
    raise SystemExit(f"unknown landscape topic {topic}")


def cmd_kubectl(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    resource = args.resource
    if resource == "nodes":
        _print_table(
            [
                {
                    "NAME": node.name,
                    "STATUS": node.status,
                    "GPU": f"{node.gpus}x{node.gpu_type}",
                    "NET": node.network,
                    "XCONN": node.interconnect,
                    "RACK": node.rack,
                }
                for node in engine.nodes.values()
            ],
            ["NAME", "STATUS", "GPU", "NET", "XCONN", "RACK"],
        )
        return 0
    if resource == "pods":
        rows = []
        for worker in engine.workers.values():
            rows.append(
                {
                    "NAME": worker.name,
                    "READY": worker.phase,
                    "NODE": worker.node or "",
                    "RANK": worker.rank,
                    "STEP": worker.step,
                }
            )
        _print_table(rows, ["NAME", "READY", "NODE", "RANK", "STEP"])
        return 0
    if resource == "jobs":
        return cmd_job_list(args)
    raise SystemExit("kubectl get nodes|pods|jobs")


def cmd_serve(args: argparse.Namespace) -> int:
    from llmtrainlab.api import run

    run(host=args.host, port=args.port, root=args.root)
    return 0


def _until(engine: ClusterEngine, predicate, limit: int = 800) -> None:
    for _ in range(limit):
        if predicate():
            return
        engine.step(1)
    raise RuntimeError(f"timeout at tick={engine.tick}")


def cmd_demo_canonical(args: argparse.Namespace) -> int:
    engine = ClusterEngine(default_cluster_spec())
    print("=== 1. 提交 Job A：6 x H100，低优先级 ===")
    engine.submit_job(yaml.safe_load((EXAMPLES / "jobs" / "job-a.yaml").read_text()))
    _until(engine, lambda: engine.jobs["llama-a"].phase == "Running")
    print(f"    A running at t={engine.tick}, H100 used={engine.gpu_inventory()['H100']['used']}")

    print("=== 2. 提交 Job B：2 x H100，高优先级（占满剩余 slot）===")
    engine.submit_job(yaml.safe_load((EXAMPLES / "jobs" / "job-b.yaml").read_text()))
    _until(engine, lambda: engine.jobs["llama-b"].phase == "Running")
    print(f"    B running at t={engine.tick}, H100 used={engine.gpu_inventory()['H100']['used']}")

    print("=== 3. 提交 Job C：4 x H100 → 进入队列 ===")
    engine.submit_job(yaml.safe_load((EXAMPLES / "jobs" / "job-c.yaml").read_text()))
    engine.step(3)
    print(f"    C phase={engine.jobs['llama-c'].phase} (期望 Queued)")

    print("=== 4. 杀掉 A 的 worker-2，整组从 checkpoint 恢复 ===")
    engine.step(engine.jobs["llama-a"].checkpoint_every + 5)
    ckpt_before = engine._latest_ckpt("llama-a")
    print(f"    checkpoint before fault: step={ckpt_before.step if ckpt_before else None}")
    engine.fault_pod("llama-a-worker-2")
    _until(engine, lambda: engine.jobs["llama-a"].phase == "Running" and engine.jobs["llama-a"].retries >= 1)
    print(
        f"    A recovered retries={engine.jobs['llama-a'].retries} "
        f"lost_steps={engine.jobs['llama-a'].lost_steps} "
        f"ckpt={engine._latest_ckpt('llama-a').step if engine._latest_ckpt('llama-a') else None}"
    )

    print("=== 5. 可选抢占：再提交 8 x H100 的高优先级 inference ===")
    if args.preempt:
        engine.submit_job(yaml.safe_load((EXAMPLES / "jobs" / "preempt-high.yaml").read_text()))
        engine.step(4)
        print(f"    urgent={engine.jobs['llama-urgent'].phase} A={engine.jobs['llama-a'].phase} C={engine.jobs['llama-c'].phase}")

    print("=== 6. KWOK 风格扩到虚拟节点，观察队列被吸干 ===")
    engine.scale_virtual_nodes(args.virtual_nodes, gpus=8, gpu_type="H100")
    engine.step(8)
    print(f"    nodes={len(engine.nodes)} C={engine.jobs['llama-c'].phase}")

    print("=== 7. p50 / p95 ===")
    metrics = engine.metrics()
    print(json.dumps({k: metrics[k] for k in [
        "tick", "gpu_slot_util", "pending_pods", "queue_wait_p50", "queue_wait_p95",
        "admit_latency_p50", "admit_latency_p95", "recovery_p50", "recovery_p95",
        "tenant_gpus",
    ]}, indent=2))
    _save(engine, args.root)
    print(f"\nstate saved to {state_path(args.root)}")
    return 0


def cmd_demo_walk(args: argparse.Namespace) -> int:
    return cmd_tutorial_print(args)


def cmd_tutorial_print(_args: argparse.Namespace) -> int:
    from llmtrainlab.tutorial import WALK

    print(WALK)
    print()
    print(
        """
# Appendix — default cluster gang / preempt / KWOK (after the landscape tutorial)

llmctl cluster init --config examples/cluster.yaml
llmctl job submit examples/jobs/job-a.yaml
llmctl tick --steps 8
llmctl job submit examples/jobs/job-b.yaml
llmctl tick --steps 4
llmctl job submit examples/jobs/job-c.yaml
llmctl tick --steps 3          # C stays Queued
llmctl tick --steps 120
llmctl fault pod llama-a-worker-2
llmctl demo canonical --preempt --virtual-nodes 40
""".strip()
    )
    return 0


def cmd_tutorial_run(args: argparse.Namespace) -> int:
    from llmtrainlab.tutorial import run_tutorial

    engine = run_tutorial()
    _save(engine, args.root)
    print("tutorial finished. expected: lora+70b-full+vllm running, ds-rlhf and bad-tp-l40s Queued, lora retries>=1")
    print(f"state → {state_path(args.root)}")
    cmd_job_list(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmctl", description="LLMTrainLab 本地训练控制面")
    parser.add_argument("--root", type=Path, default=None, help="状态目录的父路径，默认 cwd")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cluster = sub.add_parser("cluster")
    cluster_sub = cluster.add_subparsers(dest="cluster_cmd", required=True)
    init = cluster_sub.add_parser("init")
    init.add_argument("--config", type=str, default=None)
    init.set_defaults(func=cmd_cluster_init)
    status = cluster_sub.add_parser("status")
    status.set_defaults(func=cmd_cluster_status)

    job = sub.add_parser("job")
    job_sub = job.add_subparsers(dest="job_cmd", required=True)
    submit = job_sub.add_parser("submit")
    submit.add_argument("file")
    submit.add_argument("--no-tick", action="store_true")
    submit.set_defaults(func=cmd_job_submit)
    listed = job_sub.add_parser("list")
    listed.set_defaults(func=cmd_job_list)
    st = job_sub.add_parser("status")
    st.add_argument("name")
    st.set_defaults(func=cmd_job_status)

    tick = sub.add_parser("tick")
    tick.add_argument("--steps", "-n", type=int, default=1)
    tick.set_defaults(func=cmd_tick)

    fault = sub.add_parser("fault")
    fault.add_argument("kind", choices=["pod", "node", "network", "ckpt", "gpu"])
    fault.add_argument("target", nargs="?", default="")
    fault.add_argument("--job", default="")
    fault.add_argument("--duration", type=int, default=20)
    fault.add_argument("--remaining", type=int, default=0)
    fault.add_argument("--steps", type=int, default=2)
    fault.set_defaults(func=cmd_fault)

    rec = sub.add_parser("recover-node")
    rec.add_argument("name")
    rec.set_defaults(func=cmd_recover_node)

    scale = sub.add_parser("scale")
    scale.add_argument("--nodes", type=int, required=True)
    scale.add_argument("--gpus", type=int, default=8)
    scale.add_argument("--gpu-type", default="H100")
    scale.set_defaults(func=cmd_scale)

    metrics = sub.add_parser("metrics")
    metrics.add_argument("--json", action="store_true")
    metrics.set_defaults(func=cmd_metrics)

    landscape = sub.add_parser("landscape")
    landscape.add_argument(
        "topic",
        choices=["gpus", "models", "methods", "engines", "parallelism", "recipe"],
    )
    landscape.add_argument("--model", default="llama-70b")
    landscape.add_argument("--method", default="full")
    landscape.add_argument("--gpu", default=None)
    landscape.set_defaults(func=cmd_landscape)

    kubectl = sub.add_parser("kubectl")
    kubectl.add_argument("verb", choices=["get"])
    kubectl.add_argument("resource", choices=["nodes", "pods", "jobs"])
    kubectl.set_defaults(func=cmd_kubectl)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(func=cmd_serve)

    demo = sub.add_parser("demo")
    demo_sub = demo.add_subparsers(dest="demo_cmd", required=True)
    canonical = demo_sub.add_parser("canonical")
    canonical.add_argument("--virtual-nodes", type=int, default=500)
    canonical.add_argument("--preempt", action="store_true")
    canonical.set_defaults(func=cmd_demo_canonical)
    walk = demo_sub.add_parser("walk")
    walk.set_defaults(func=cmd_demo_walk)

    tutorial = sub.add_parser("tutorial")
    tutorial_sub = tutorial.add_subparsers(dest="tutorial_cmd", required=True)
    t_print = tutorial_sub.add_parser("print")
    t_print.set_defaults(func=cmd_tutorial_print)
    t_run = tutorial_sub.add_parser("run")
    t_run.set_defaults(func=cmd_tutorial_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"unknown object: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
