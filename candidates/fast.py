"""묶음 학습: 설정이 같고 초기화 난수만 다른 실행 여러 개를 한 번에 학습한다 (순환형 후보용).

Folio · Strata · Lattice 는 토큰마다 작은 GPU 연산을 수백 번 부른다. 그래서 계산량보다 호출 비용이 시간을 정한다.
  1) torch.func.vmap 으로 N개 모델의 가중치를 쌓아 같은 연산 한 번에 N개를 계산한다 (작은 연산이라 시간은 거의 그대로).
  2) 청크 하나의 forward + backward 를 CUDA graph 로 캡처해 재생한다 (파이썬 · 커널 호출 비용을 없앤다).

benchmark.train_run 과 같은 것을 계산한다: 같은 초기화(난수별 torch.manual_seed), 같은 배치 순서(공통 data_seed),
같은 손실(청크별 역전파의 합), 모델별 기울기 자르기, 같은 AdamW · 학습률 일정, 같은 평가. 다른 점은 vmap 의 부동소수
계산 순서뿐이다(기울기 차이 약 1e-9). 결과 파일(result.json · model.pt)도 실행마다 같은 형식으로 남긴다.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, stack_module_state, vmap

from .benchmark import DATA_CACHE, OUT, RunConfig, _batches, evaluate
from .models import VARIANTS, build, n_params
from .tasks import SPECS, VOCAB, dataset

FAST_VARIANTS = ("folio", "strata", "lattice")   # 기본 모델 이름 앞부분. Transformer · GRU 는 원래 빠르다
TRAINER = "grouped-graph-v1"
# 모델 1개 · 배치 64 · 청크 64 학습의 활성값 메모리 (GB, measure.json 의 피크 근처). 청크에 비례한다
ACT_GB = {"folio": 0.25, "strata": 0.08, "lattice": 1.3}


def segments(cfg: RunConfig, n_models: int, budget_gb: float) -> int:
    """활성값이 예산을 넘으면 청크를 시간으로 k 조각 낸다. 앞으로는 저장 없이 조각 경계 상태만 남기고,
    뒤에서부터 조각마다 다시 계산해 역전파한다(손으로 한 활성값 체크포인트). 기울기는 청크 전체 BPTT 와 같다."""
    need = n_models * ACT_GB[VARIANTS[cfg.variant][0]] * cfg.batch / 64 * cfg.chunk / 64
    k = 1
    while need / k > budget_gb and k < cfg.chunk:
        k *= 2
    return k


def gpu_budget_gb(workers: int) -> float:
    """프로세스 하나가 쓸 활성값 예산. GPU 메모리를 넘으면 Windows 가 시스템 메모리로 넘겨 매우 느려진다."""
    total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 8.0
    return max(0.3, (total - 1.0 - 0.5 * workers) / workers)


def eligible(cfg: RunConfig) -> bool:
    return VARIANTS[cfg.variant][0] in FAST_VARIANTS and all(SPECS[t].length % cfg.chunk == 0 for t in cfg.tasks)


def group_configs(configs: list[RunConfig]) -> list[list[RunConfig]]:
    """난수만 다른 설정끼리 묶는다. 묶을 수 없는 설정은 하나짜리 묶음."""
    groups: dict[str, list[RunConfig]] = {}
    for c in configs:
        k = json.dumps(asdict(replace(c, seed=0)), sort_keys=True, default=str) if eligible(c) else c.run_id()
        groups.setdefault(k, []).append(c)
    return list(groups.values())


class _MathGRUCell(nn.GRUCell):
    """nn.GRUCell 과 같은 식을 일반 연산으로. vmap 에 GRUCell 일괄 규칙이 없어 모델마다 따로 돌던 것을 한 번에 계산한다."""

    def forward(self, x, h):
        gi = F.linear(x, self.weight_ih, self.bias_ih)
        gh = F.linear(h, self.weight_hh, self.bias_hh)
        ir, iz, inn = gi.chunk(3, -1)
        hr, hz, hn = gh.chunk(3, -1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        n = torch.tanh(inn + r * hn)
        return (1 - z) * n + z * h


class GroupTrainer:
    def __init__(self, cfgs: list[RunConfig], device: str = "cuda", budget_gb: float = 2.0):
        base_cfg = cfgs[0]
        assert all(replace(c, seed=0) == replace(base_cfg, seed=0) for c in cfgs), "난수만 달라야 한다"
        assert eligible(base_cfg)
        self.cfgs, self.cfg, self.dev = cfgs, base_cfg, torch.device(device)
        self.N = len(cfgs)
        models = []
        for c in cfgs:                                         # train_run 과 같은 초기화
            torch.manual_seed(c.seed)
            models.append(build(c.variant, VOCAB).to(self.dev))
        self.template = models[0]
        params, bufs = stack_module_state(models)
        self.params = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
        self.bufs = {k: v.detach().clone() for k, v in bufs.items()}
        base = copy.deepcopy(models[0]).to("meta")
        for m in base.modules():
            if type(m) is nn.GRUCell:
                m.__class__ = _MathGRUCell
        base.masked_path = True                                # Strata: 시계를 CPU 로 읽지 않는다
        self.base = base
        self.state_keys = list(models[0].initial_state(1, self.dev))
        self.aux_w = getattr(models[0].cfg, "aux_weight", 0.0)
        self.graph = None
        self.k = segments(base_cfg, self.N, budget_gb)
        assert self.k == 1 or self.aux_w == 0, "보조 손실(Strata)은 청크 단위로 평균하므로 조각 내지 않는다"

    # ------------------------------------------------------------ 청크 하나: N개 모델의 기울기
    def _chunk_fn(self):
        base, keys, aux_w, L = self.base, self.state_keys, self.aux_w, SPECS[self.cfg.tasks[0]].length

        def loss(p, b, tok, tgt, mask, counts, *state):
            logits, st = functional_call(base, (p, b), (tok, dict(zip(keys, state))))
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
            total = ((ce * mask).sum(dim=1) / counts).mean()
            if aux_w > 0:
                aux = base.aux_loss()
                if aux is not None:
                    total = total + aux_w * aux * tok.shape[1] / L
            return total, tuple(st[k] for k in keys)

        n_state = len(keys)
        return vmap(grad(loss, has_aux=True), in_dims=(0, 0, None, None, None, None) + (0,) * n_state)

    def _init_state(self, B):
        st = self.template.initial_state(B, self.dev)
        return [v.expand(self.N, *v.shape).contiguous() for v in st.values()]

    def _capture(self, B):
        C = self.cfg.chunk
        self.s_tok = torch.zeros(B, C, dtype=torch.long, device=self.dev)
        self.s_tgt = torch.zeros(B, C, dtype=torch.long, device=self.dev)
        self.s_mask = torch.zeros(B, C, device=self.dev)
        self.s_cnt = torch.ones(B, device=self.dev)
        self.s_state = self._init_state(B)
        f = self._chunk_fn()
        args = lambda: (self.params, self.bufs, self.s_tok, self.s_tgt, self.s_mask, self.s_cnt, *self.s_state)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(side):
            for _ in range(2):
                f(*args())
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.g_out, self.st_out = f(*args())

    def step_grads(self, tok, sc):
        """배치 하나의 기울기를 params[k].grad 에 둔다 (모든 청크의 합)."""
        if self.k > 1:
            return self._step_seg(tok, sc)
        B, Ltot = tok.shape
        C = self.cfg.chunk
        if self.graph is None:
            self._capture(B)
        targets = torch.cat([tok[:, 1:], tok[:, :1]], dim=1)
        counts = sc.float().sum(dim=1).clamp(min=1)
        acc = {k: torch.zeros_like(v) for k, v in self.params.items()}
        init = self._init_state(B)
        for s, v in zip(self.s_state, init):
            s.copy_(v)
        self.s_cnt.copy_(counts)
        for c0 in range(0, Ltot, C):
            if self.cfg.memory == "window" and c0 > 0:
                for s, v in zip(self.s_state, init):
                    s.copy_(v)
            self.s_tok.copy_(tok[:, c0:c0 + C])
            self.s_tgt.copy_(targets[:, c0:c0 + C])
            self.s_mask.copy_(sc[:, c0:c0 + C].float())
            self.graph.replay()
            torch._foreach_add_(list(acc.values()), [self.g_out[k] for k in acc])
            for s, o in zip(self.s_state, self.st_out):
                s.copy_(o)
        for k, p in self.params.items():
            p.grad = acc[k]

    # ------------------------------------------------------------ 청크를 시간으로 조각 내는 경로 (메모리가 모자랄 때)
    def _seg_fns(self):
        """조각 하나: 앞으로(상태만) · 뒤로(가중치 기울기, 들어온 상태의 기울기)."""
        base, keys = self.base, self.state_keys

        def run(p, b, tok, *state):
            return functional_call(base, (p, b), (tok, dict(zip(keys, state))))

        def fwd(p, b, tok, *state):
            return tuple(run(p, b, tok, *state)[1][k] for k in keys)

        def bwd(p, b, tok, tgt, mask, counts, gout, *state):
            def f(p_, *s):
                logits, st = run(p_, b, tok, *s)
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
                return ((ce * mask).sum(dim=1) / counts).mean(), tuple(st[k] for k in keys)
            (loss, _), vjp_fn = torch.func.vjp(f, p, *state)
            g = vjp_fn((torch.ones_like(loss), tuple(gout)))
            return g[0], tuple(g[1:])

        n = len(keys)
        return (vmap(fwd, in_dims=(0, 0, None) + (0,) * n),
                vmap(bwd, in_dims=(0, 0, None, None, None, None, 0) + (0,) * n))

    def _capture_seg(self, B, S):
        self.s_tok = torch.zeros(B, S, dtype=torch.long, device=self.dev)
        self.s_tgt = torch.zeros(B, S, dtype=torch.long, device=self.dev)
        self.s_mask = torch.zeros(B, S, device=self.dev)
        self.s_cnt = torch.ones(B, device=self.dev)
        self.s_state = self._init_state(B)
        self.s_gout = [torch.zeros_like(v) for v in self.s_state]
        fwd, bwd = self._seg_fns()
        fa = lambda: (self.params, self.bufs, self.s_tok, *self.s_state)
        ba = lambda: (self.params, self.bufs, self.s_tok, self.s_tgt, self.s_mask, self.s_cnt, tuple(self.s_gout), *self.s_state)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(side):
            for _ in range(2):
                fwd(*fa()), bwd(*ba())
        torch.cuda.current_stream().wait_stream(side)
        self.graph_f, self.graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph_f):
            self.f_out = fwd(*fa())
        with torch.no_grad(), torch.cuda.graph(self.graph, pool=self.graph_f.pool()):
            self.g_out, self.gin_out = bwd(*ba())

    def _step_seg(self, tok, sc):
        B, Ltot = tok.shape
        C = self.cfg.chunk
        S = C // self.k
        if self.graph is None:
            self._capture_seg(B, S)
        targets = torch.cat([tok[:, 1:], tok[:, :1]], dim=1)
        self.s_cnt.copy_(sc.float().sum(dim=1).clamp(min=1))
        acc = {k: torch.zeros_like(v) for k, v in self.params.items()}
        init = self._init_state(B)
        cur = [v.clone() for v in init]
        for c0 in range(0, Ltot, C):
            if self.cfg.memory == "window" and c0 > 0:
                cur = [v.clone() for v in init]
            starts = []
            for j in range(self.k):                                   # 앞으로: 조각 경계 상태만 남긴다
                starts.append(cur)
                for s, v in zip(self.s_state, cur):
                    s.copy_(v)
                self.s_tok.copy_(tok[:, c0 + j * S:c0 + (j + 1) * S])
                self.graph_f.replay()
                cur = [o.clone() for o in self.f_out]
            for g in self.s_gout:                                      # 청크 끝 상태로는 기울기가 오지 않는다 (detach)
                g.zero_()
            for j in reversed(range(self.k)):                          # 뒤로: 조각마다 다시 계산해 역전파
                sl = slice(c0 + j * S, c0 + (j + 1) * S)
                for s, v in zip(self.s_state, starts[j]):
                    s.copy_(v)
                self.s_tok.copy_(tok[:, sl])
                self.s_tgt.copy_(targets[:, sl])
                self.s_mask.copy_(sc[:, sl].float())
                self.graph.replay()
                torch._foreach_add_(list(acc.values()), [self.g_out[k] for k in acc])
                gin = [g.clone() for g in self.gin_out]
                for g, v in zip(self.s_gout, gin):
                    g.copy_(v)
        for k, p in self.params.items():
            p.grad = acc[k]

    def clip_(self, max_norm=1.0):
        """모델별 clip_grad_norm_ (전체 노름은 모델마다 따로)."""
        sq = sum(p.grad.pow(2).reshape(self.N, -1).sum(dim=1) for p in self.params.values())
        coef = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
        for p in self.params.values():
            p.grad.mul_(coef.view(-1, *([1] * (p.dim() - 1))))

    def model(self, i: int) -> nn.Module:
        m = build(self.cfgs[i].variant, VOCAB).to(self.dev)
        m.load_state_dict({k: v[i].detach() for k, v in (self.params | self.bufs).items()}, strict=False)
        return m


def train_group(cfgs: list[RunConfig], device: str = "cuda", log=print, budget_gb: float = 2.0) -> list[dict]:
    """benchmark.train_run 과 같은 결과 파일을 실행마다 남긴다. 이미 끝난 실행은 건너뛴다."""
    todo = []
    for c in cfgs:
        rp = OUT / "runs" / c.run_id() / "result.json"
        if not (rp.exists() and json.loads(rp.read_text(encoding="utf-8"))["config_key"] == json.loads(json.dumps(c.key(), default=str))):
            todo.append(c)
    if not todo:
        return [json.loads((OUT / "runs" / c.run_id() / "result.json").read_text(encoding="utf-8")) for c in cfgs]
    cfg = todo[0]
    tr = GroupTrainer(todo, device, budget_gb)
    dev = tr.dev
    gid = hashlib.sha1("|".join(c.run_id() for c in todo).encode()).hexdigest()[:10]
    gdir = OUT / "groups" / gid
    gdir.mkdir(parents=True, exist_ok=True)
    for c in todo:
        rd = OUT / "runs" / c.run_id()
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "config.json").write_text(json.dumps(c.key(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    names = ", ".join(c.run_id() for c in todo)

    train = {t: dataset(t, "train", cfg.n_train, DATA_CACHE) for t in cfg.tasks}
    val = {t: dataset(t, "val", cfg.n_val, DATA_CACHE) for t in cfg.tasks}
    plist = list(tr.params.values())
    opt = torch.optim.AdamW(plist, lr=cfg.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps))))
    start, elapsed, hist = 0, 0.0, [[] for _ in todo]
    ckpt = gdir / "ckpt.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        with torch.no_grad():
            for k, v in st["params"].items():
                tr.params[k].copy_(v)
        opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        start, elapsed, hist = st["step"], st["elapsed"], st["history"]
        log(f"  묶음 {gid}: {start}걸음부터 이어서")
    gen = _batches(cfg)
    for _ in range(start):
        next(gen)
    log(f"  묶음 {gid} ({len(todo)}개 동시{f', 청크를 {tr.k}조각으로 나눠 역전파' if tr.k > 1 else ''}): {names}")
    t0 = time.perf_counter()
    for step in range(start, cfg.steps):
        rows = next(gen)
        tok = torch.from_numpy(np.stack([train[t]["tokens"][i] for t, i in rows])).to(dev)
        sc = torch.from_numpy(np.stack([train[t]["score"][i] for t, i in rows])).to(dev)
        tr.step_grads(tok, sc)
        tr.clip_(1.0)
        opt.step()
        sched.step()
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            torch.cuda.synchronize()
            now = elapsed + time.perf_counter() - t0
            for i, c in enumerate(todo):
                m = tr.model(i)
                v = {t: evaluate(m, val[t], cfg.memory, cfg.chunk, dev) for t in cfg.tasks}
                hist[i].append({"step": step + 1, "seconds": round(now, 1),
                                "val": {t: {"accuracy": round(x["accuracy"], 4), "bits": round(x["bits"], 4)} for t, x in v.items()}})
                log(f"  {c.run_id()} {step + 1}/{cfg.steps}  val acc " + " ".join(f"{t}:{x['accuracy']:.3f}" for t, x in v.items()))
            torch.save({"params": {k: v.detach() for k, v in tr.params.items()}, "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1, "elapsed": now, "history": hist}, ckpt)
    torch.cuda.synchronize()
    elapsed += time.perf_counter() - t0
    out = []
    for i, c in enumerate(todo):
        m = tr.model(i)
        test = {t: evaluate(m, dataset(t, "test", c.n_test if SPECS[t].length <= 256 else c.n_test // 2, DATA_CACHE),
                            c.memory, c.chunk, dev) for t in c.all_eval_tasks()}
        rd = OUT / "runs" / c.run_id()
        res = {"run_id": c.run_id(), "config_key": json.loads(json.dumps(c.key(), default=str)),
               "variant": c.variant, "mode": c.mode(), "tasks": list(c.tasks), "memory": c.memory, "seed": c.seed,
               "params": n_params(m), "train_tokens": c.steps * c.batch * SPECS[c.tasks[0]].length,
               "wall_seconds": round(elapsed, 1), "trainer": TRAINER, "group_size": len(todo), "history": hist[i], "test": test}
        (rd / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        torch.save(m.state_dict(), rd / "model.pt")
        out.append(res)
    ckpt.unlink(missing_ok=True)
    return out
