"""Tick-based cluster: gang admission, queues, fake workers, checkpoints, faults."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

from llmtrainlab.landscape import GPU_TYPES, GPUS, apply_recipe, fabric_step_cost


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


@dataclass
class GPU:
    gpu_id: str
    gpu_type: str
    node: str
    allocated_to: str | None = None
    available: bool = True


@dataclass
class Node:
    name: str
    gpu_type: str
    gpus: int
    zone: str = "az-a"
    rack: str = "rack-1"
    network: str = "rdma"
    interconnect: str = "nvlink"
    status: str = "Ready"  # Ready | NotReady | Faulted
    kind: str = "physical"  # physical | kwok


@dataclass
class Team:
    name: str
    namespace: str
    guaranteed_gpus: int
    burst_gpus: int
    priority: int


@dataclass
class Worker:
    name: str
    job: str
    rank: int
    world_size: int
    phase: str = "Pending"
    node: str | None = None
    gpu_ids: list[str] = field(default_factory=list)
    step: int = 0
    last_heartbeat: int = 0
    bound_tick: int | None = None


@dataclass
class Checkpoint:
    job: str
    step: int
    tick: int
    intact: bool = True


@dataclass
class Job:
    name: str
    namespace: str
    team: str
    priority: int
    workers: int
    gpus_per_worker: int
    gpu_type: str
    steps: int
    checkpoint_every: int
    require_rdma: bool = False
    prefer_same_rack: bool = True
    qos: str = "training"
    model: str = ""
    method: str = "full"
    engine: str = ""
    dp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    prefer_same_node: bool = False
    fabric_cost: int = 0
    hbm_gb_needed: float = 0.0
    phase: str = "Queued"
    arrival_tick: int = 0
    admit_tick: int | None = None
    start_train_tick: int | None = None
    finish_tick: int | None = None
    retries: int = 0
    lost_steps: int = 0
    detection_ticks: list[int] = field(default_factory=list)
    recovery_ticks: list[int] = field(default_factory=list)
    recover_started_tick: int | None = None
    fail_detected_tick: int | None = None
    network_extra_ticks: int = 0
    progress_budget: int = 0  # extra ticks owed per step
    events: list[str] = field(default_factory=list)

    @property
    def gpu_demand(self) -> int:
        return self.workers * self.gpus_per_worker


@dataclass
class MetricSample:
    wait: list[float] = field(default_factory=list)
    recovery: list[float] = field(default_factory=list)
    admit_latency: list[float] = field(default_factory=list)


class ClusterEngine:
    def __init__(self, spec: dict[str, Any] | None = None):
        spec = spec or {}
        self.tick = 0
        self.heartbeat_timeout = 3
        self.checkpoint_store_down_until = -1
        self.nodes: dict[str, Node] = {}
        self.gpus: dict[str, GPU] = {}
        self.teams: dict[str, Team] = {}
        self.jobs: dict[str, Job] = {}
        self.workers: dict[str, Worker] = {}
        self.checkpoints: dict[str, list[Checkpoint]] = {}
        self.samples = MetricSample()
        self.log: list[str] = []
        self._load_spec(spec)

    def _note(self, message: str) -> None:
        self.log.append(f"t={self.tick} {message}")

    def _load_spec(self, spec: dict[str, Any]) -> None:
        for raw in spec.get("nodes", []):
            self.add_node(raw)
        for name, raw in spec.get("teams", {}).items():
            self.teams[name] = Team(
                name=name,
                namespace=raw.get("namespace", name),
                guaranteed_gpus=int(raw.get("guaranteed_gpus", 0)),
                burst_gpus=int(raw.get("burst_gpus", raw.get("guaranteed_gpus", 0))),
                priority=int(raw.get("priority", 50)),
            )
        if not self.teams:
            self.teams = {
                "team-a": Team("team-a", "team-a", 6, 10, 80),
                "team-b": Team("team-b", "team-b", 2, 4, 20),
                "team-c": Team("team-c", "team-c", 4, 8, 100),
            }

    def add_node(self, raw: dict[str, Any]) -> Node:
        node = Node(
            name=str(raw["name"]),
            gpu_type=str(raw["gpu_type"]),
            gpus=int(raw["gpus"]),
            zone=str(raw.get("zone", "az-a")),
            rack=str(raw.get("rack", "rack-1")),
            network=str(raw.get("network", "rdma")),
            interconnect=str(raw.get("interconnect", "nvlink")),
            status=str(raw.get("status", "Ready")),
            kind=str(raw.get("kind", "physical")),
        )
        self.nodes[node.name] = node
        for index in range(node.gpus):
            gpu_id = f"{node.name}-gpu-{index}"
            self.gpus[gpu_id] = GPU(gpu_id, node.gpu_type, node.name)
        return node

    def snapshot(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "heartbeat_timeout": self.heartbeat_timeout,
            "checkpoint_store_down_until": self.checkpoint_store_down_until,
            "nodes": {name: asdict(node) for name, node in self.nodes.items()},
            "gpus": {name: asdict(gpu) for name, gpu in self.gpus.items()},
            "teams": {name: asdict(team) for name, team in self.teams.items()},
            "jobs": {name: asdict(job) for name, job in self.jobs.items()},
            "workers": {name: asdict(worker) for name, worker in self.workers.items()},
            "checkpoints": {
                job: [asdict(item) for item in items]
                for job, items in self.checkpoints.items()
            },
            "samples": asdict(self.samples),
            "log": self.log[-400:],
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> ClusterEngine:
        engine = cls({"teams": {}})
        engine.tick = int(data.get("tick", 0))
        engine.heartbeat_timeout = int(data.get("heartbeat_timeout", 3))
        engine.checkpoint_store_down_until = int(data.get("checkpoint_store_down_until", -1))
        engine.nodes = {k: Node(**v) for k, v in data.get("nodes", {}).items()}
        engine.gpus = {k: GPU(**v) for k, v in data.get("gpus", {}).items()}
        engine.teams = {k: Team(**v) for k, v in data.get("teams", {}).items()}
        job_fields = Job.__dataclass_fields__
        engine.jobs = {}
        for name, raw in data.get("jobs", {}).items():
            engine.jobs[name] = Job(**{key: raw[key] for key in raw if key in job_fields})
        engine.workers = {k: Worker(**v) for k, v in data.get("workers", {}).items()}
        engine.checkpoints = {
            job: [Checkpoint(**item) for item in items]
            for job, items in data.get("checkpoints", {}).items()
        }
        samples = data.get("samples", {})
        engine.samples = MetricSample(
            wait=list(samples.get("wait", [])),
            recovery=list(samples.get("recovery", [])),
            admit_latency=list(samples.get("admit_latency", [])),
        )
        engine.log = list(data.get("log", []))
        return engine

    def submit_job(self, raw: dict[str, Any]) -> Job:
        meta = raw.get("metadata", {})
        spec = dict(raw.get("spec", raw))
        spec = apply_recipe(spec)
        name = str(meta.get("name") or spec["name"])
        if name in self.jobs:
            raise ValueError(f"job {name} already exists")
        par = spec.get("parallelism") or {}
        dp = int(par.get("dp") or spec.get("dp") or 1)
        tp = int(par.get("tp") or spec.get("tp") or 1)
        pp = int(par.get("pp") or spec.get("pp") or 1)
        ep = int(par.get("ep") or spec.get("ep") or 1)
        workers = int(spec.get("workers") or dp * tp * pp * ep)
        job = Job(
            name=name,
            namespace=str(meta.get("namespace") or spec.get("namespace") or spec.get("team")),
            team=str(spec["team"]),
            priority=int(spec.get("priority", self.teams.get(spec["team"], Team(spec["team"], spec["team"], 0, 0, 50)).priority)),
            workers=workers,
            gpus_per_worker=int(spec.get("gpusPerWorker") or spec.get("gpus_per_worker") or 1),
            gpu_type=str(spec.get("gpuType") or spec.get("gpu_type")),
            steps=int(spec.get("steps", 100)),
            checkpoint_every=int(spec.get("checkpointEvery") or spec.get("checkpoint_every") or 20),
            require_rdma=bool(spec.get("requireRdma") or spec.get("require_rdma") or False),
            prefer_same_rack=bool(spec.get("preferSameRack", spec.get("prefer_same_rack", True))),
            prefer_same_node=bool(spec.get("preferSameNode") or spec.get("prefer_same_node") or False),
            qos=str(spec.get("qos", "training")),
            model=str(spec.get("model") or ""),
            method=str(spec.get("method") or "full"),
            engine=str(spec.get("engine") or ""),
            dp=dp,
            tp=tp,
            pp=pp,
            ep=ep,
            hbm_gb_needed=float(spec.get("_hbm_gb_needed") or 0.0),
            arrival_tick=self.tick,
        )
        if job.gpu_type not in GPU_TYPES:
            raise ValueError(f"unknown gpuType {job.gpu_type}")
        if job.ep > 1:
            job.require_rdma = True
        if job.tp > 1:
            job.prefer_same_node = True
        self.jobs[name] = job
        reasons = spec.get("_recipe_reasons") or []
        if reasons:
            job.events.extend(reasons)
        self._note(f"submit {name} demand={job.gpu_demand} {job.gpu_type} pri={job.priority} ns={job.namespace} {job.method} tp={job.tp} ep={job.ep}")
        return job

    def _node(self, name: str) -> Node:
        return self.nodes[name]

    def _free_gpus(self, job: Job) -> list[GPU]:
        free: list[GPU] = []
        for gpu in self.gpus.values():
            if gpu.allocated_to or not gpu.available:
                continue
            node = self._node(gpu.node)
            if node.status != "Ready":
                continue
            if gpu.gpu_type != job.gpu_type:
                continue
            sku = GPUS[gpu.gpu_type]
            if job.hbm_gb_needed and sku.hbm_gb + 1e-6 < job.hbm_gb_needed:
                continue
            if job.require_rdma and node.network != "rdma":
                continue
            if job.tp > 1 and node.interconnect != "nvlink":
                continue
            free.append(gpu)
        return free

    def _score_pack(self, job: Job, gpus: list[GPU]) -> list[GPU] | None:
        need = job.gpu_demand
        if len(gpus) < need:
            return None
        by_node: dict[str, list[GPU]] = {}
        by_rack: dict[str, list[GPU]] = {}
        for gpu in gpus:
            by_node.setdefault(gpu.node, []).append(gpu)
            by_rack.setdefault(self._node(gpu.node).rack, []).append(gpu)
        if job.prefer_same_node:
            ordered_nodes = sorted(by_node, key=lambda name: (-len(by_node[name]), name))
            if len(by_node[ordered_nodes[0]]) >= need:
                return by_node[ordered_nodes[0]][:need]
        ordered_racks = sorted(by_rack, key=lambda rack: (-len(by_rack[rack]), rack))
        chosen: list[GPU] = []
        if job.prefer_same_rack and len(by_rack[ordered_racks[0]]) >= need:
            chosen = by_rack[ordered_racks[0]][:need]
        else:
            for rack in ordered_racks:
                for gpu in by_rack[rack]:
                    chosen.append(gpu)
                    if len(chosen) == need:
                        break
                if len(chosen) == need:
                    break
        return chosen if len(chosen) == need else None

    def try_place(self, job: Job) -> list[GPU] | None:
        return self._score_pack(job, self._free_gpus(job))

    def _team_usage(self, team: str) -> int:
        used = 0
        for worker in self.workers.values():
            job = self.jobs[worker.job]
            if job.team == team and worker.phase not in {"Failed", "Stopped"}:
                used += len(worker.gpu_ids) or job.gpus_per_worker
        return used

    def _fair_key(self, job: Job) -> tuple:
        team = self.teams.get(job.team)
        guaranteed = max(1, team.guaranteed_gpus if team else 1)
        share = self._team_usage(job.team) / guaranteed
        burst = team.burst_gpus if team else 10**9
        over_burst = 1 if self._team_usage(job.team) + job.gpu_demand > burst else 0
        return (-job.priority, over_burst, share, job.arrival_tick, job.name)

    def _queue_order(self) -> list[Job]:
        return sorted(
            (job for job in self.jobs.values() if job.phase in {"Queued", "Preempted", "Recovering"}),
            key=self._fair_key,
        )

    def _release_worker(self, worker: Worker) -> None:
        for gpu_id in worker.gpu_ids:
            gpu = self.gpus[gpu_id]
            gpu.allocated_to = None
        worker.gpu_ids = []
        worker.node = None
        worker.phase = "Stopped"

    def _bind_job(self, job: Job, gpus: list[GPU]) -> None:
        job.phase = "Starting"
        if job.admit_tick is None:
            job.admit_tick = self.tick
            wait = float(self.tick - job.arrival_tick)
            self.samples.wait.append(wait)
            self.samples.admit_latency.append(wait)
        per = job.gpus_per_worker
        existing = [w for w in self.workers.values() if w.job == job.name]
        for worker in existing:
            self._release_worker(worker)
            self.workers.pop(worker.name, None)
        for rank in range(job.workers):
            slice_ = gpus[rank * per : (rank + 1) * per]
            worker = Worker(
                name=f"{job.name}-worker-{rank}",
                job=job.name,
                rank=rank,
                world_size=job.workers,
                phase="Bound",
                node=slice_[0].node,
                gpu_ids=[gpu.gpu_id for gpu in slice_],
                bound_tick=self.tick,
                last_heartbeat=self.tick,
            )
            for gpu in slice_:
                gpu.allocated_to = worker.name
            self.workers[worker.name] = worker
        racks = {self._node(gpu.node).rack for gpu in gpus}
        nodes = [self._node(gpu.node) for gpu in gpus]
        job.fabric_cost = fabric_step_cost(
            job.tp,
            job.ep,
            job.dp,
            (node.interconnect for node in nodes),
            (node.network for node in nodes),
            extra=job.network_extra_ticks,
        )
        job.events.append(f"t={self.tick} placed workers={job.workers} racks={sorted(racks)} fabric_cost={job.fabric_cost}")
        self._note(f"admit {job.name} workers={job.workers} racks={sorted(racks)}")

    def _stop_job_workers(self, job: Job, reason: str) -> None:
        for worker in list(self.workers.values()):
            if worker.job != job.name:
                continue
            self._release_worker(worker)
        job.events.append(f"t={self.tick} stop {reason}")

    def _latest_ckpt(self, job_name: str) -> Checkpoint | None:
        items = [item for item in self.checkpoints.get(job_name, []) if item.intact]
        return items[-1] if items else None

    def _preempt(self, job: Job) -> bool:
        victims = sorted(
            (
                other
                for other in self.jobs.values()
                if other.phase in {"Starting", "Running"} and other.priority < job.priority
            ),
            key=lambda item: (item.priority, -item.gpu_demand, item.name),
        )
        if not victims:
            return False
        reserved: list[Job] = []
        snapshot_gpus = deepcopy(self.gpus)
        snapshot_workers = deepcopy(self.workers)
        snapshot_jobs = {name: deepcopy(job_) for name, job_ in self.jobs.items()}
        for victim in victims:
            self._stop_job_workers(victim, f"preempted-by-{job.name}")
            victim.phase = "Preempted"
            reserved.append(victim)
            if self.try_place(job):
                for item in reserved:
                    item.events.append(f"t={self.tick} preempted by {job.name}")
                    self._note(f"preempt {item.name} -> queued, winner={job.name}")
                return True
        self.gpus = snapshot_gpus
        self.workers = snapshot_workers
        for name, snap in snapshot_jobs.items():
            live = self.jobs[name]
            for field_name in Job.__dataclass_fields__:
                setattr(live, field_name, getattr(snap, field_name))
        return False

    def _admit_loop(self) -> None:
        names = [item.name for item in self._queue_order()]
        for name in names:
            job = self.jobs[name]
            if job.phase not in {"Queued", "Preempted", "Recovering"}:
                continue
            if job.phase == "Recovering" and any(
                worker.job == job.name and worker.phase not in {"Stopped", "Failed"}
                for worker in self.workers.values()
            ):
                continue
            team = self.teams.get(job.team)
            if (
                team
                and job.phase != "Recovering"
                and self._team_usage(job.team) + job.gpu_demand > team.burst_gpus
            ):
                continue
            placement = self.try_place(job)
            if placement is None and job.phase in {"Queued", "Preempted"}:
                if self._preempt(job):
                    placement = self.try_place(job)
            if placement is None:
                continue
            if job.phase == "Recovering":
                ckpt = self._latest_ckpt(job.name)
                restore = ckpt.step if ckpt else 0
                for worker in list(self.workers.values()):
                    if worker.job == job.name:
                        self.workers.pop(worker.name, None)
                if job.fail_detected_tick is not None:
                    recovered = int(self.tick - job.fail_detected_tick)
                    job.recovery_ticks.append(recovered)
                    self.samples.recovery.append(float(recovered))
                job.fail_detected_tick = None
                job.recover_started_tick = None
                self._bind_job(job, placement)
                for worker in self.workers.values():
                    if worker.job == job.name:
                        worker.step = restore
                job.phase = "Starting"
                self._note(f"recover {job.name} from ckpt={restore}")
                continue
            self._bind_job(job, placement)

    def _maybe_checkpoint(self, job: Job, step: int) -> None:
        if step <= 0 or step % job.checkpoint_every != 0:
            return
        intact = self.tick >= self.checkpoint_store_down_until
        record = Checkpoint(job=job.name, step=step, tick=self.tick, intact=intact)
        self.checkpoints.setdefault(job.name, []).append(record)
        job.events.append(f"t={self.tick} checkpoint step={step} intact={intact}")

    def _advance_workers(self) -> None:
        by_job: dict[str, list[Worker]] = {}
        for worker in self.workers.values():
            by_job.setdefault(worker.job, []).append(worker)

        for job_name, group in by_job.items():
            job = self.jobs[job_name]
            if job.phase in {"Succeeded", "Queued", "Preempted"}:
                continue

            for worker in group:
                if worker.phase == "Bound" and worker.bound_tick is not None and self.tick > worker.bound_tick:
                    worker.phase = "Running"
                    worker.last_heartbeat = self.tick

            live = [worker for worker in group if worker.phase not in {"Failed", "Stopped"}]
            if (
                job.phase in {"Starting", "Running"}
                and len(live) == job.workers
                and all(worker.phase in {"Running", "Ready", "Training"} for worker in live)
            ):
                first_rendezvous = job.start_train_tick is None
                for worker in live:
                    worker.phase = "Training"
                    worker.last_heartbeat = self.tick
                job.phase = "Running"
                if first_rendezvous:
                    job.start_train_tick = self.tick
                    self._note(f"rendezvous {job.name} world_size={job.workers}")

            if job.phase != "Running":
                continue

            step_cost = max(1, job.fabric_cost)
            if job.network_extra_ticks:
                step_cost = max(step_cost, job.network_extra_ticks)
            if step_cost > 1:
                job.progress_budget += 1
                if job.progress_budget < step_cost:
                    for worker in live:
                        worker.last_heartbeat = self.tick
                    continue
                job.progress_budget = 0

            steps_now = []
            for worker in live:
                worker.step += 1
                worker.last_heartbeat = self.tick
                steps_now.append(worker.step)
            synced = min(steps_now) if steps_now else 0
            self._maybe_checkpoint(job, synced)
            if synced >= job.steps:
                job.phase = "Succeeded"
                job.finish_tick = self.tick
                self._stop_job_workers(job, "succeeded")
                self._note(f"succeed {job.name} steps={synced}")

    def _detect_failures(self) -> None:
        failed_jobs: set[str] = set()
        for worker in list(self.workers.values()):
            if worker.phase == "Stopped":
                continue
            if worker.phase != "Failed":
                silent = self.tick - worker.last_heartbeat > self.heartbeat_timeout
                if silent:
                    worker.phase = "Failed"
                    self._note(f"heartbeat-lost {worker.name}")
            if worker.phase == "Failed":
                failed_jobs.add(worker.job)

        for job_name in failed_jobs:
            job = self.jobs[job_name]
            if job.phase in {"Succeeded", "Recovering"}:
                continue
            if job.fail_detected_tick is None:
                job.fail_detected_tick = self.tick
                job.detection_ticks.append(self.heartbeat_timeout)
            current = max((item.step for item in self.workers.values() if item.job == job.name), default=0)
            ckpt = self._latest_ckpt(job.name)
            restore = ckpt.step if ckpt else 0
            job.lost_steps += max(0, current - restore)
            job.retries += 1
            job.recover_started_tick = self.tick
            job.phase = "Recovering"
            self._stop_job_workers(job, f"worker-failed:{job_name}")
            self._note(f"fail-group {job.name} lost_steps+={max(0, current - restore)} ckpt={restore}")

    def step(self, n: int = 1) -> None:
        for _ in range(max(0, n)):
            self.tick += 1
            self._detect_failures()
            self._admit_loop()
            self._advance_workers()

    def fault_pod(self, worker_name: str) -> Worker:
        worker = self.workers[worker_name]
        worker.phase = "Failed"
        worker.last_heartbeat = self.tick - self.heartbeat_timeout - 1
        self._note(f"inject pod-fault {worker_name}")
        return worker

    def fault_node(self, node_name: str) -> None:
        node = self.nodes[node_name]
        node.status = "Faulted"
        for gpu in self.gpus.values():
            if gpu.node == node_name:
                gpu.available = False
        for worker in self.workers.values():
            if worker.node == node_name and worker.phase not in {"Stopped"}:
                worker.phase = "Failed"
                worker.last_heartbeat = self.tick - self.heartbeat_timeout - 1
        self._note(f"inject node-fault {node_name}")

    def recover_node(self, node_name: str) -> None:
        node = self.nodes[node_name]
        node.status = "Ready"
        for gpu in self.gpus.values():
            if gpu.node == node_name:
                gpu.available = True
        self._note(f"recover-node {node_name}")

    def fault_network(self, job_name: str, extra_ticks: int = 3) -> None:
        job = self.jobs[job_name]
        job.network_extra_ticks = extra_ticks
        self._note(f"inject network-delay {job_name} extra_ticks={extra_ticks}")

    def fault_checkpoint_store(self, duration: int = 20) -> None:
        self.checkpoint_store_down_until = self.tick + duration
        self._note(f"inject ckpt-store-down until t={self.checkpoint_store_down_until}")

    def shrink_gpus(self, node_name: str, remaining: int) -> None:
        node = self.nodes[node_name]
        kept = 0
        for gpu in self.gpus.values():
            if gpu.node != node_name:
                continue
            if kept < remaining and gpu.allocated_to is None:
                kept += 1
                continue
            if gpu.allocated_to is None:
                gpu.available = False
        node.gpus = remaining
        self._note(f"inject gpu-shrink {node_name} remaining={remaining}")

    def scale_virtual_nodes(self, count: int, gpus: int = 8, gpu_type: str = "H100") -> int:
        created = 0
        existing = {name for name in self.nodes if name.startswith("kwok-")}
        for index in range(count):
            name = f"kwok-{index:03d}"
            if name in existing:
                continue
            self.add_node(
                {
                    "name": name,
                    "gpu_type": gpu_type,
                    "gpus": gpus,
                    "zone": "az-sim",
                    "rack": f"rack-sim-{index % 20}",
                    "network": "rdma",
                    "interconnect": "nvlink",
                    "kind": "kwok",
                }
            )
            created += 1
        self._note(f"scale kwok nodes +{created} (requested {count})")
        return created

    def gpu_inventory(self) -> dict[str, Any]:
        by_type: dict[str, dict[str, int]] = {}
        for gpu in self.gpus.values():
            bucket = by_type.setdefault(gpu.gpu_type, {"total": 0, "allocatable": 0, "used": 0})
            bucket["total"] += 1
            node = self._node(gpu.node)
            if gpu.available and node.status == "Ready":
                bucket["allocatable"] += 1
            if gpu.allocated_to:
                bucket["used"] += 1
        return by_type

    def pending_pods(self) -> int:
        queued = sum(job.workers for job in self.jobs.values() if job.phase in {"Queued", "Preempted", "Recovering"})
        pending_workers = sum(1 for worker in self.workers.values() if worker.phase in {"Pending", "Bound"})
        return queued + pending_workers

    def metrics(self) -> dict[str, Any]:
        inv = self.gpu_inventory()
        used = sum(item["used"] for item in inv.values())
        allocatable = sum(item["allocatable"] for item in inv.values()) or 1
        tenants = {name: self._team_usage(name) for name in self.teams}
        ckpt_age = {}
        for job in self.jobs.values():
            ckpt = self._latest_ckpt(job.name)
            ckpt_age[job.name] = (self.tick - ckpt.tick) if ckpt else None
        through = [
            max(worker.step for worker in self.workers.values() if worker.job == job.name) / max(1, self.tick)
            for job in self.jobs.values()
            if any(worker.job == job.name for worker in self.workers.values())
        ]
        return {
            "tick": self.tick,
            "gpu_slot_util": round(used / allocatable, 4),
            "gpu_inventory": inv,
            "pending_pods": self.pending_pods(),
            "tenant_gpus": tenants,
            "checkpoint_age": ckpt_age,
            "queue_wait_p50": _pctl(self.samples.wait, 0.5),
            "queue_wait_p95": _pctl(self.samples.wait, 0.95),
            "admit_latency_p50": _pctl(self.samples.admit_latency, 0.5),
            "admit_latency_p95": _pctl(self.samples.admit_latency, 0.95),
            "recovery_p50": _pctl(self.samples.recovery, 0.5),
            "recovery_p95": _pctl(self.samples.recovery, 0.95),
            "recovery_samples": list(self.samples.recovery),
            "controller_reconcile_latency": 1,
            "training_throughput": round(median(through), 4) if through else 0.0,
            "dcgm": {
                name: {
                    "sku": node.gpu_type,
                    "hbm_gb": GPUS[node.gpu_type].hbm_gb,
                    "hbm_tbs": GPUS[node.gpu_type].hbm_tbs,
                    "nvlink": GPUS[node.gpu_type].nvlink,
                    "sm_util": round(used_on / max(1, node.gpus), 4),
                    "nvlink_active": node.interconnect == "nvlink" and used_on > 1,
                }
                for name, node in self.nodes.items()
                for used_on in [sum(1 for gpu in self.gpus.values() if gpu.node == name and gpu.allocated_to)]
            },
            "jobs": {
                name: {
                    "phase": job.phase,
                    "retries": job.retries,
                    "lost_steps": job.lost_steps,
                    "priority": job.priority,
                    "qos": job.qos,
                    "model": job.model,
                    "method": job.method,
                    "engine": job.engine,
                    "parallelism": {"dp": job.dp, "tp": job.tp, "pp": job.pp, "ep": job.ep},
                    "fabric_cost": job.fabric_cost,
                    "checkpoint": (self._latest_ckpt(name).step if self._latest_ckpt(name) else None),
                    "workers": [
                        {
                            "name": worker.name,
                            "rank": worker.rank,
                            "phase": worker.phase,
                            "node": worker.node,
                            "step": worker.step,
                        }
                        for worker in self.workers.values()
                        if worker.job == name
                    ],
                }
                for name, job in self.jobs.items()
            },
        }


def default_cluster_spec() -> dict[str, Any]:
    return {
        "nodes": [
            {"name": "node-a", "gpu_type": "A100", "gpus": 4, "zone": "az-a", "rack": "rack-1", "network": "ethernet", "interconnect": "nvlink"},
            {"name": "node-b", "gpu_type": "H100", "gpus": 8, "zone": "az-a", "rack": "rack-1", "network": "rdma", "interconnect": "nvlink"},
            {"name": "node-c", "gpu_type": "L40S", "gpus": 2, "zone": "az-b", "rack": "rack-2", "network": "ethernet", "interconnect": "pcie"},
        ],
        "teams": {
            "team-a": {"namespace": "team-a", "guaranteed_gpus": 6, "burst_gpus": 10, "priority": 80},
            "team-b": {"namespace": "team-b", "guaranteed_gpus": 2, "burst_gpus": 4, "priority": 20},
            "team-c": {"namespace": "team-c", "guaranteed_gpus": 4, "burst_gpus": 8, "priority": 100},
        },
    }
