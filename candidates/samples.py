"""'직접 풀기' 화면용 자료: 실제 시험 문제와, 저장된 모델이 그 문제에 낸 실제 답 → samples.json

재학습은 없고 model.pt 가 필요하다. 문제는 모델 점수를 보지 않고 아래 규칙으로만 고른다(시험 세트 앞쪽부터).
답을 뽑는 경로로 전체 시험을 다시 채점해서, 기록된 정확도와 다르면 멈춘다.

    py -m candidates.samples
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .benchmark import DATA_CACHE, OUT
from .models import build, detach_state
from .tasks import ARROW, SEP, VOCAB, dataset, delay_of

N_QUESTIONS = 8
MARK_COPY, MARK_REV = 29, 30
OPEN_A, OPEN_B = 29, 30
QUERY = 25
SOURCES = {"kv": "E", "chain": "E", "paren": "H", "copy": "H"}


def _kv(tokens, score, n):
    """앞쪽 14회 안의 질의 중, 답이 나온 지 세 묶음(9기호) 넘게 지난 것. 에피소드마다 하나."""
    out = []
    for e, (t, s) in enumerate(zip(tokens, score)):
        delays = delay_of(t, s)
        for p in np.flatnonzero(s):
            if 16 <= p <= 40 and delays[p] >= 9:
                out.append((e, 0, int(p)))
                break
        if len(out) == n:
            break
    return out


def _chain(tokens, score, n):
    """사슬을 두 번 이상 따라가야 답이 나오는 질의. 에피소드마다 하나. 보여 주는 것은 그 블록."""
    out = []
    for e, (t, s) in enumerate(zip(tokens, score)):
        for p in np.flatnonzero(s):
            start = int(np.flatnonzero(t[:p + 1] == SEP)[-1])
            facts = {int(t[i - 1]): int(t[i + 1]) for i in range(start + 1, p) if t[i] == ARROW}
            hops, u = 0, int(t[p])
            while u in facts:
                u, hops = facts[u], hops + 1
            if hops >= 2:
                out.append((e, start, int(p)))
                break
        if len(out) == n:
            break
    return out


def _paren(tokens, score, n):
    """앞쪽 20–90기호에서 깊이 2 이상일 때 닫는 자리. 에피소드마다 하나."""
    out = []
    for e, (t, s) in enumerate(zip(tokens, score)):
        depth = 0
        for i in range(len(t) - 1):
            if t[i] in (OPEN_A, OPEN_B):
                depth += 1
            elif t[i] == QUERY:
                if s[i] and 20 <= i <= 90 and depth >= 2:
                    out.append((e, 0, i))
                    break
                depth -= 1
        if len(out) == n:
            break
    return out


def _copy(tokens, score, n):
    """원문 10–20기호 블록의 중간 자리. 그대로 · 거꾸로 절반씩. 에피소드마다 하나."""
    out, quota = [], {MARK_COPY: n - n // 2, MARK_REV: n // 2}
    for e, t in enumerate(tokens):
        for start in np.flatnonzero(t == SEP):
            marks = np.flatnonzero((t[start + 1:] == MARK_COPY) | (t[start + 1:] == MARK_REV))
            if not len(marks):
                break
            j = int(start + 1 + marks[0])
            size = j - start - 1
            if 10 <= size <= 20 and quota[int(t[j])] > 0:
                quota[int(t[j])] -= 1
                out.append((e, int(start), j + size // 2))
                break
        if len(out) == n:
            break
    return out


PICK = {"kv": _kv, "chain": _chain, "paren": _paren, "copy": _copy}


@torch.no_grad()
def predict(model, tokens: np.ndarray, memory: str, chunk: int, device, batch: int = 64) -> np.ndarray:
    """위치별 예측 (B, L). delay.hits_per_position 과 같은 순서로 상태를 넘긴다."""
    model.eval()
    out = []
    for i in range(0, len(tokens), batch):
        t = torch.from_numpy(tokens[i:i + batch]).to(device)
        B, L = t.shape
        state, rows = None, []
        for c0 in range(0, L, chunk):
            if memory == "window" or state is None:
                state = model.initial_state(B, device)
            logits, state = model(t[:, c0:c0 + chunk], state)
            state = detach_state(state)
            rows.append(logits.argmax(-1).cpu().numpy())
        out.append(np.concatenate(rows, axis=1))
    return np.concatenate(out, axis=0)


def _runs(group: str) -> list[dict]:
    first = {"E": "switch_fast", "H": "cyc"}[group]
    out = []
    for p in sorted((OUT / "runs").glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        cfg = r["config_key"]
        if r["tasks"][0] == first and cfg["steps"] == 3000 and r["mode"] == "joint" and r["memory"] == "persistent" \
                and p.with_name("model.pt").exists() and r["variant"] != "strata_sameperiod":
            out.append({**r, "path": p})
    return sorted(out, key=lambda r: (r["variant"], r["seed"]))


def main(device: str = "cuda"):
    out = {"questions_per_task": N_QUESTIONS, "tasks": {}}
    for task, group in SOURCES.items():
        runs = _runs(group)
        d = dataset(task, "test", runs[0]["config_key"]["n_test"], DATA_CACHE)
        tok, sc = d["tokens"], d["score"]
        picks = PICK[task](tok, sc, N_QUESTIONS)
        if len(picks) != N_QUESTIONS:
            raise SystemExit(f"{task}: 문제 {len(picks)}개뿐")
        targets = np.concatenate([tok[:, 1:], tok[:, :1]], axis=1)
        preds = {}
        for r in runs:
            model = build(r["variant"], VOCAB).to(device)
            model.load_state_dict(torch.load(r["path"].with_name("model.pt"), map_location=device, weights_only=True))
            cfg = r["config_key"]
            pr = predict(model, tok, r["memory"], cfg["chunk"], torch.device(device))
            acc = float((pr == targets)[sc].mean())
            want = r["test"][task]["accuracy"]
            if abs(acc - want) > 0.003:
                raise SystemExit(f"{r['run_id']} {task}: 다시 채점 {acc:.4f}, 기록 {want:.4f}")
            print(f"{r['variant']:>15} s{r['seed']} {task:>6} 다시 채점 {acc:.4f} / 기록 {want:.4f}", flush=True)
            preds.setdefault(r["variant"], []).append([int(pr[e, p]) for e, _, p in picks])
            del model
        out["tasks"][task] = {
            "group": group,
            "questions": [{"prompt": tok[e, s:p + 1].tolist(), "answer": int(targets[e, p]),
                           "preds": {v: [seed[q] for seed in seeds] for v, seeds in preds.items()}}
                          for q, (e, s, p) in enumerate(picks)],
        }
    path = OUT / "samples.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"저장: {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
