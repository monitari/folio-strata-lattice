"""학습·평가·실행 관리·자원 측정.

기억 조건 (ARCHITECTURE_CANDIDATES.md §6)
  window     : 모든 모델의 상태를 chunk(=64) 토큰마다 초기화한다. 모든 모델이 같은 창 안의 정보만 본다.
  persistent : 에피소드 안에서 상태를 이어 간다 (학습은 truncated BPTT, 청크 사이 detach).
               Transformer 의 상태는 최근 context 토큰 창이므로 그보다 오래된 정보는 볼 수 없다.
               Transformer 는 두 조건 모두 청크마다 새로 시작해 학습한다 (창을 넘는 학습 경로가 없다).
학습 모드
  joint    : 여러 과제를 한 배치에 섞어 한 모델로 학습 (행 i 의 과제 = i mod 과제 수)
  separate : 과제 하나로만 학습
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models import VARIANTS, build, describe, detach_state, n_params, state_bytes
from .tasks import SPECS, SUITE, TRAIN_TASKS, VOCAB, data_fingerprint, dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "candidates"
DATA_CACHE = OUT / "data"
CODE_VERSION = "cand-train-v1"


@dataclass
class RunConfig:
    variant: str
    tasks: tuple                      # 학습 과제
    memory: str = "persistent"        # window | persistent
    seed: int = 0                     # 모델 초기화 난수 (데이터 순서와 별개)
    steps: int = 1500
    batch: int = 32
    chunk: int = 64                   # BPTT 길이 = Transformer 문맥 길이
    lr: float = 2e-3
    warmup: int = 100
    n_train: int = 2048               # 과제별 학습 에피소드 수
    n_val: int = 128
    n_test: int = 256
    eval_every: int = 250
    data_seed: int = 12345            # 데이터 순서 (모든 모델 공통)
    eval_tasks: tuple = ()            # 비우면 학습 과제 + 그 과제의 긴 변형

    def mode(self) -> str:
        return "joint" if len(self.tasks) > 1 else "separate"

    def all_eval_tasks(self) -> tuple:
        if self.eval_tasks:
            return tuple(self.eval_tasks)
        longs = [k for k, s in SPECS.items() if not s.train and k.replace("_long", "") in self.tasks]
        return tuple(self.tasks) + tuple(longs)

    def key(self) -> dict:
        d = asdict(self) | {"model": describe(build(self.variant, VOCAB))["config"], "data": data_fingerprint(tuple(self.tasks) + self.all_eval_tasks()),
                            "code": CODE_VERSION, "eval": self.all_eval_tasks()}
        d["tasks"], d["eval_tasks"] = list(self.tasks), list(self.eval_tasks)
        return d

    def run_id(self) -> str:
        h = hashlib.sha1(json.dumps(self.key(), sort_keys=True, default=str).encode()).hexdigest()[:8]
        tag = "suite" if tuple(self.tasks) == SUITE else "joint" if self.mode() == "joint" else self.tasks[0]
        return f"{self.variant}-{tag}-{self.memory}-s{self.seed}-{h}"


# ---------------------------------------------------------------- 손실과 평가
def _row_loss(logits, targets, mask):
    """행(에피소드)별 채점 위치의 CE 합. logits (B,T,V), targets/mask (B,T)."""
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view_as(targets)
    return (ce * mask).sum(dim=1)


def run_episodes(model, tokens, score, memory: str, chunk: int, grad: bool = False, row_counts=None, aux_w: float = 0.0):
    """에피소드 묶음을 청크 단위로 처리한다. grad=True 면 청크마다 역전파(가중치 갱신은 하지 않음).

    반환: (행별 CE 합, 행별 정답 수, 마지막 청크의 보조 손실 합) — 평가에는 앞의 둘을 쓴다.
    """
    B, L = tokens.shape
    dev = tokens.device
    targets = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1)          # 마지막 위치는 score=False
    state = None
    ce_sum = torch.zeros(B, device=dev)
    hits = torch.zeros(B, device=dev)
    is_tf = type(model).__name__ == "TransformerBaseline"
    for c0 in range(0, L, chunk):
        sl = slice(c0, min(L, c0 + chunk))
        if memory == "window" or state is None or (grad and is_tf):
            state = model.initial_state(B, dev)
        logits, state = model(tokens[:, sl], state)
        m = score[:, sl].float()
        row = _row_loss(logits, targets[:, sl], m)
        if grad:
            loss = (row / row_counts).mean()
            aux = model.aux_loss()
            if aux is not None and aux_w > 0:
                loss = loss + aux_w * aux * (sl.stop - sl.start) / L     # 청크 길이만큼 나눠 더한다
            loss.backward()
            state = detach_state(state)
        ce_sum += row.detach()
        hits += ((logits.argmax(-1) == targets[:, sl]).float() * m).sum(dim=1).detach()
    return ce_sum, hits


@torch.no_grad()
def evaluate(model, data: dict, memory: str, chunk: int, device, batch: int = 128) -> dict:
    model.eval()
    tok, sc = torch.from_numpy(data["tokens"]), torch.from_numpy(data["score"])
    ce, hit, n = 0.0, 0.0, 0.0
    for i in range(0, len(tok), batch):
        t, s = tok[i:i + batch].to(device), sc[i:i + batch].to(device)
        c, h = run_episodes(model, t, s, memory, chunk)
        ce, hit, n = ce + float(c.sum()), hit + float(h.sum()), n + float(s.sum())
    model.train()
    targets = np.concatenate([data["tokens"][:, 1:], data["tokens"][:, :1]], axis=1)
    orc = float((data["oracle"][data["score"]] == targets[data["score"]]).mean())
    return {"accuracy": hit / n, "bits": ce / n / math.log(2), "scored": int(n), "oracle_accuracy": orc}


# ---------------------------------------------------------------- 한 실행
def _batches(cfg: RunConfig):
    """모든 모델에 같은 배치 순서. 행 i 의 과제 = tasks[i mod 과제 수]."""
    rng = np.random.default_rng(cfg.data_seed)
    orders = {t: [] for t in cfg.tasks}
    while True:
        rows = []
        for i in range(cfg.batch):
            t = cfg.tasks[i % len(cfg.tasks)]
            if not orders[t]:
                orders[t] = list(rng.permutation(cfg.n_train))
            rows.append((t, orders[t].pop()))
        yield rows


def train_run(cfg: RunConfig, device: str = "cuda", log=print) -> dict:
    run_dir = OUT / "runs" / cfg.run_id()
    result_path = run_dir / "result.json"
    if result_path.exists():
        res = json.loads(result_path.read_text(encoding="utf-8"))
        if res["config_key"] == json.loads(json.dumps(cfg.key(), default=str)):
            return res                                          # 같은 설정의 완료된 결과만 재사용
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.key(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    dev = torch.device(device)
    train = {t: dataset(t, "train", cfg.n_train, DATA_CACHE) for t in cfg.tasks}
    val = {t: dataset(t, "val", cfg.n_val, DATA_CACHE) for t in cfg.tasks}
    torch.manual_seed(cfg.seed)
    model = build(cfg.variant, VOCAB).to(dev)
    # 트랜스포머는 학습 때 청크마다 창을 비운다. 청크가 창보다 짧으면 창 뒤쪽 위치 임베딩을 한 번도 배우지 않는다.
    assert type(model).__name__ != "TransformerBaseline" or cfg.chunk >= model.cfg.context, "트랜스포머 청크 < 창"
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps))))
    aux_w = getattr(model.cfg, "aux_weight", 0.0)
    start_step, elapsed, history = 0, 0.0, []
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():                                          # 중단된 실행 이어서
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        start_step, elapsed, history = st["step"], st["elapsed"], st["history"]
        log(f"  {cfg.run_id()}: {start_step}걸음부터 이어서")
    gen = _batches(cfg)
    for _ in range(start_step):                                # 같은 배치 순서를 유지하기 위해 앞부분을 건너뛴다
        next(gen)
    tokens_seen = 0
    t0 = time.perf_counter()
    for step in range(start_step, cfg.steps):
        rows = next(gen)
        tok = torch.from_numpy(np.stack([train[t]["tokens"][i] for t, i in rows])).to(dev)
        sc = torch.from_numpy(np.stack([train[t]["score"][i] for t, i in rows])).to(dev)
        counts = sc.float().sum(dim=1).clamp(min=1)
        opt.zero_grad(set_to_none=True)
        run_episodes(model, tok, sc, cfg.memory, cfg.chunk, grad=True, row_counts=counts, aux_w=aux_w)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tokens_seen += tok.numel()
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            now = elapsed + time.perf_counter() - t0
            v = {t: evaluate(model, val[t], cfg.memory, cfg.chunk, dev) for t in cfg.tasks}
            history.append({"step": step + 1, "seconds": round(now, 1),
                            "val": {t: {"accuracy": round(x["accuracy"], 4), "bits": round(x["bits"], 4)} for t, x in v.items()}})
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1, "elapsed": now, "history": history}, ckpt)
            log(f"  {cfg.run_id()} {step + 1}/{cfg.steps}  val acc " + " ".join(f"{t}:{x['accuracy']:.3f}" for t, x in v.items()))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed += time.perf_counter() - t0

    test = {t: evaluate(model, dataset(t, "test", cfg.n_test if SPECS[t].length <= 256 else cfg.n_test // 2, DATA_CACHE),
                        cfg.memory, cfg.chunk, dev) for t in cfg.all_eval_tasks()}
    res = {"run_id": cfg.run_id(), "config_key": json.loads(json.dumps(cfg.key(), default=str)),
           "variant": cfg.variant, "mode": cfg.mode(), "tasks": list(cfg.tasks), "memory": cfg.memory, "seed": cfg.seed,
           "params": n_params(model), "train_tokens": cfg.steps * cfg.batch * SPECS[cfg.tasks[0]].length,
           "wall_seconds": round(elapsed, 1), "history": history, "test": test}
    result_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "model.pt")
    ckpt.unlink(missing_ok=True)
    return res


def prepare_data(configs: list[RunConfig]):
    for c in configs:
        for t in c.tasks:
            dataset(t, "train", c.n_train, DATA_CACHE)
            dataset(t, "val", c.n_val, DATA_CACHE)
        for t in c.all_eval_tasks():
            dataset(t, "test", c.n_test if SPECS[t].length <= 256 else c.n_test // 2, DATA_CACHE)


def _run_job(cfgs: list[RunConfig], device: str, log, budget_gb: float = 2.0) -> list[dict]:
    """순환형 후보는 GPU 에서 난수 묶음으로 한 번에 학습한다(fast.py). 나머지는 한 실행씩."""
    from .fast import eligible, train_group
    if device == "cuda" and eligible(cfgs[0]):
        return train_group(cfgs, device, log, budget_gb)
    return [train_run(c, device, log) for c in cfgs]


def _worker(args):
    cfg_dicts, device, budget = args
    import warnings
    warnings.filterwarnings("ignore", message=".*AccumulateGrad.*|.*batching rule.*")
    return _run_job([RunConfig(**d) for d in cfg_dicts], device, lambda m: print(m, flush=True), budget)


def run_many(configs: list[RunConfig], workers: int = 1, device: str = "cuda", log=print) -> list[dict]:
    """여러 실행. workers>1 이면 프로세스를 나눠 GPU 를 함께 쓴다 (이때 wall_seconds 는 비교에 쓰지 않는다).
    난수만 다른 순환형 후보 실행은 한 일감으로 묶는다. 오래 걸리는 일감부터 돌린다."""
    from .fast import group_configs
    jobs = group_configs(configs)
    jobs.sort(key=lambda g: -g[0].steps * SEC_WEIGHT.get(VARIANTS[g[0].variant][0], 1.0))
    from .fast import gpu_budget_gb
    budget = gpu_budget_gb(max(1, workers)) if device == "cuda" else 0
    if workers <= 1:
        return [r for g in jobs for r in _run_job(g, device, log, budget)]
    prepare_data(configs)   # 병렬 프로세스가 같은 캐시를 동시에 만들지 않게 미리 한 번 만든다
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        return [r for rs in pool.map(_worker, [([asdict(c) for c in g], device, budget) for g in jobs]) for r in rs]


SEC_WEIGHT = {"lattice": 3.0, "strata": 1.2, "folio": 1.0, "gru": 0.1, "transformer": 0.1}   # 일감 순서용 상대 비용


# ---------------------------------------------------------------- 자원 측정 (단독 실행에서만)
def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def measure(variant: str, device: str = "cuda", batch: int = 32, length: int = 256, chunk: int = 64,
            train_steps: int = 20, warmup: int = 5, latency_tokens: int = 256) -> dict:
    """학습 처리량·피크 메모리·배치 1 온라인 지연·지속 상태 크기. 동기화와 준비 실행을 거친 값이다."""
    dev = torch.device(device)
    torch.manual_seed(0)
    model = build(variant, VOCAB).to(dev)
    chunk = max(chunk, getattr(model.cfg, "context", 0))   # 트랜스포머는 청크 = 창 (train_run 과 같게)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = dataset("kv", "train", batch, DATA_CACHE)
    tok = torch.from_numpy(data["tokens"][:batch, :length]).to(dev)
    sc = torch.from_numpy(data["score"][:batch, :length]).to(dev)
    counts = sc.float().sum(dim=1).clamp(min=1)

    def one():
        opt.zero_grad(set_to_none=True)
        run_episodes(model, tok, sc, "persistent", chunk, grad=True, row_counts=counts, aux_w=getattr(model.cfg, "aux_weight", 0.0))
        opt.step()

    for _ in range(warmup):
        one()
    _sync(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
    t0 = time.perf_counter()
    for _ in range(train_steps):
        one()
    _sync(dev)
    sec = (time.perf_counter() - t0) / train_steps
    peak = (torch.cuda.max_memory_allocated(dev) - base) if dev.type == "cuda" else None

    model.eval()
    lat = {}
    for ldev in ([dev, torch.device("cpu")] if dev.type == "cuda" else [dev]):
        m = build(variant, VOCAB).to(ldev).eval()
        m.load_state_dict({k: v.to(ldev) for k, v in model.state_dict().items()})
        with torch.no_grad():
            st = m.initial_state(1, ldev)
            x = torch.zeros(1, dtype=torch.long, device=ldev)
            for _ in range(chunk + 8):                       # 준비 실행: Transformer 는 창을 꽉 채운 상태에서 잰다
                _, st = m.step(x, st)
            _sync(ldev)
            times = []
            for _ in range(latency_tokens):
                t1 = time.perf_counter()
                _, st = m.step(x, st)
                _sync(ldev)
                times.append(time.perf_counter() - t1)
        lat[ldev.type] = {"mean_ms": 1000 * float(np.mean(times)), "p90_ms": 1000 * float(np.percentile(times, 90))}
    return {"variant": variant, "params": n_params(model), "train_sec_per_step": sec,
            "train_tokens_per_sec": batch * length / sec, "train_peak_bytes": peak,
            "state_bytes_batch1": state_bytes(model.initial_state(1, dev)), "latency": lat,
            "setup": {"batch": batch, "length": length, "chunk": chunk, "memory": "persistent", "device": device}}
