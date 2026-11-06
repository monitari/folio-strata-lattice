"""구현 검증 (ARCHITECTURE_CANDIDATES.md §6 '구현 검증'). 모든 변형 모델에 같은 검사를 한다.

    py -m candidates.checks              # CPU, float64 로 정밀 검사
    py -m candidates.checks --device cuda

1. forward 와 step 누적 결과 일치
2. 미래 입력을 바꿔도 과거 예측 불변 (인과성)
3. 에피소드 사이 상태 누출 없음: 같은 배치의 다른 행·이전 에피소드가 결과에 영향을 주지 않는다
4. 청크 분할 전후 상태·시계·예측 일치 (평가 경로, detach 는 값에 영향 없음)
5. 유한한 손실, 유효한 gradient (사용하지 않기로 한 대조군 가중치는 제외)
6. 작은 결정적 문제의 학습 가능성
7. 생성 과제의 정답 정확성
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .models import VARIANTS, build, detach_state
from .tasks import (SPECS, VOCAB, check_bracket_depth, check_copy_reverse, check_count_mod, check_cyclic_chain, check_key_value,
                    check_relation_chain, episode)

TOL = 1e-9          # float64 기준
UNUSED = {"folio_nomem": ("keys", "w_read", "w_write", "w_erase", "w_value"),
          "lattice_nonbr": ("w_dir",)}


def _model(variant, device):
    torch.manual_seed(0)
    return build(variant, VOCAB).to(device).double().eval()


def _close(a, b):
    return float((a - b).abs().max())


def check_forward_vs_step(v, dev):
    m = _model(v, dev)
    worst = 0.0
    for T in (40, 80):                                           # Transformer: 창(64) 안의 한 번 계산 경로와 창이 밀리는 경로
        x = torch.randint(0, VOCAB, (3, T), device=dev)
        with torch.no_grad():
            lf, sf = m(x)
            st, outs = m.initial_state(3, dev), []
            for t in range(T):
                lo, st = m.step(x[:, t], st)
                outs.append(lo)
            ls = torch.stack(outs, 1)
        worst = max(worst, _close(lf, ls), *(_close(sf[k].double(), st[k].double()) for k in sf))
    return worst < 1e-6, f"최대 차이 {worst:.1e}"


def check_causality(v, dev):
    m = _model(v, dev)
    x = torch.randint(0, VOCAB, (2, 70), device=dev)
    y = x.clone()
    y[:, 40:] = torch.randint(0, VOCAB, (2, 30), device=dev)       # 40번 이후만 바꾼다
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(y)
    d = _close(a[:, :40], b[:, :40])
    return d < TOL, f"과거 예측 변화 {d:.1e}"


def check_no_leak(v, dev):
    m = _model(v, dev)
    a = torch.randint(0, VOCAB, (1, 50), device=dev)
    others = torch.randint(0, VOCAB, (3, 50), device=dev)
    with torch.no_grad():
        alone, _ = m(a)
        batched, _ = m(torch.cat([a, others]))                   # 같은 배치의 다른 행
        _, st = m(others[:1])                                    # 이전 에피소드를 처리한 뒤
        fresh, _ = m(a, m.initial_state(1, dev))                 # 새 에피소드는 초기 상태에서 시작
    d = max(_close(alone[0], batched[0]), _close(alone, fresh))
    return d < 1e-8, f"차이 {d:.1e}"


def check_chunking(v, dev):
    m = _model(v, dev)
    x = torch.randint(0, VOCAB, (2, 150), device=dev)            # Strata 64 주기 경계와 어긋나게 자른다
    with torch.no_grad():
        full, sf = m(x)
        parts, st = [], None
        for a, b in ((0, 37), (37, 101), (101, 150)):
            lo, st = m(x[:, a:b], st)
            st = detach_state(st)                                # 학습 경로와 같은 detach
            parts.append(lo)
        chunked = torch.cat(parts, 1)
    d = max(_close(full, chunked), *(_close(sf[k].double(), st[k].double()) for k in sf))
    extra = f", 시계 {int(st['t'][0])}" if "t" in st else ""
    return d < 1e-6, f"최대 차이 {d:.1e}{extra}"


def check_gradients(v, dev):
    torch.manual_seed(0)
    m = build(v, VOCAB).to(dev).train()
    # 140: Strata 의 64 주기 수준이 두 번 갱신되는 길이. 첫 갱신은 상태가 0이라 weight_hh 의 gradient 가 수학적으로 0이다.
    x = torch.randint(0, VOCAB, (4, 140), device=dev)
    logits, _ = m(x[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    aux = m.aux_loss()
    if aux is not None:
        loss = loss + 0.05 * aux
    loss.backward()
    skip = UNUSED.get(v, ())
    bad = [n for n, p in m.named_parameters()
           if not n.startswith(skip) and (p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0)]
    ok = bool(torch.isfinite(loss)) and not bad
    return ok, f"손실 {loss.item():.3f}" + (f", 문제 있는 gradient: {bad}" if bad else "")


def check_learnable(v, dev, steps=300):
    """결정적 문제: 길이 7짜리 고정 순서의 반복. 다음 기호를 거의 완벽히 맞혀야 한다."""
    torch.manual_seed(0)
    m = build(v, VOCAB).to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    seq = torch.tensor([3, 1, 4, 1, 5, 9, 2], device=dev)
    x = seq.repeat(8, 6)[:, :40]
    for _ in range(steps):
        logits, _ = m(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits, _ = m(x[:, :-1])
    acc = float((logits.argmax(-1)[:, 8:] == x[:, 9:]).float().mean())     # 앞부분(주기를 보기 전) 제외
    return acc > 0.95, f"정답률 {acc:.3f}, 손실 {loss.item():.3f}"


def check_task_generators():
    rows = []
    for key, spec in SPECS.items():
        eps = [episode(key, s, i) for s in ("train", "val", "test") for i in range(20)]
        ok = all(not e.score[-1] and len(e.tokens) == spec.length and e.tokens.max() < VOCAB for e in eps)
        if key.startswith("kv"):
            ok &= all(check_key_value(e.tokens, e.score) for e in eps)
        if key.startswith("chain"):
            ok &= all(check_relation_chain(e.tokens, e.score) for e in eps)
        for prefix, fn in (("cyc", check_cyclic_chain), ("count", check_count_mod), ("copy", check_copy_reverse), ("paren", check_bracket_depth)):
            if key.split("_")[0] == prefix:
                ok &= all(fn(e.tokens, e.score) for e in eps)
        if "knob" in eps[0].meta:                                   # 난이도 값은 채점 위치에만
            ok &= all(((e.meta["knob"] >= 0) <= e.score).all() for e in eps)
        if spec.kind == "query":
            ok &= all((e.oracle[e.score] == e.tokens[1:][e.score[:-1]]).all() for e in eps)
        rows.append((key, ok))
    return all(ok for _, ok in rows), ", ".join(f"{k}{'' if ok else ' 실패'}" for k, ok in rows)


CHECKS = [("forward = step 누적", check_forward_vs_step), ("인과성", check_causality), ("상태 누출 없음", check_no_leak),
          ("청크 분할 일치", check_chunking), ("유한 손실·gradient", check_gradients)]


def run_all(device: str = "cpu", variants=VARIANTS, learn: bool = True, out=None) -> int:
    """모든 검사를 하고 실패 개수를 돌려준다. out 이 있으면 결과를 JSON 으로 남긴다 (웹 화면용)."""
    dev = torch.device(device)
    rows = []
    ok, msg = check_task_generators()
    rows.append({"variant": "", "check": "과제 생성기 정답 검증", "ok": bool(ok), "msg": msg})
    print(f"[{'통과' if ok else '실패'}] 과제 생성기 정답 검증: {msg}")
    for v in variants:
        for name, fn in CHECKS + ([("작은 결정적 문제 학습", check_learnable)] if learn else []):
            ok, msg = fn(v, dev)
            rows.append({"variant": v, "check": name, "ok": bool(ok), "msg": msg})
            print(f"[{'통과' if ok else '실패'}] {v:18} {name}: {msg}", flush=True)
    failed = sum(not r["ok"] for r in rows)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"checked_at": time.strftime("%Y-%m-%d %H:%M"), "device": device, "dtype": "float64",
                                   "learn": learn, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n실패 {failed}개")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--skip-learn", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from .benchmark import OUT
    full = args.variants == ",".join(VARIANTS) and not args.skip_learn and args.device == "cpu"
    failed = run_all(args.device, args.variants.split(","), not args.skip_learn, OUT / "checks.json" if full else None)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
