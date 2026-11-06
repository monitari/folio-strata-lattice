"""지연 사다리 키-값의 지연 구간별 정확도 → runs/<id>/delay.json (재학습 없음, 저장된 모델로 평가)

한 실행의 전체 정확도는 짧은 지연이 대부분이라 긴 지연의 성적을 가린다. 그래서 채점 위치마다 지연을 다시 계산해
구간별로 나눈다. 기준선 '가장 최근 저장값' 도 같은 구간으로 낸다.

    py -m candidates.delay
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .benchmark import DATA_CACHE, OUT
from .models import build, detach_state
from .tasks import SPECS, STORE, VOCAB, dataset, delay_of

DELAY_VERSION = "delay-v1"
BUCKETS = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
TASKS = ("kvd", "kvd_long")


def bucket_name(lo, hi):
    return f"{lo}-{hi}"


@torch.no_grad()
def hits_per_position(model, tokens: np.ndarray, memory: str, chunk: int, device, batch: int = 64) -> np.ndarray:
    """위치별 정답 여부 (B, L). benchmark.run_episodes 의 평가 경로와 같은 순서로 상태를 넘긴다."""
    model.eval()
    out = []
    for i in range(0, len(tokens), batch):
        t = torch.from_numpy(tokens[i:i + batch]).to(device)
        B, L = t.shape
        targets = torch.cat([t[:, 1:], t[:, :1]], dim=1)
        state, rows = None, []
        for c0 in range(0, L, chunk):
            sl = slice(c0, min(L, c0 + chunk))
            if memory == "window" or state is None:
                state = model.initial_state(B, device)
            logits, state = model(t[:, sl], state)
            state = detach_state(state)
            rows.append((logits.argmax(-1) == targets[:, sl]).cpu().numpy())
        out.append(np.concatenate(rows, axis=1))
    return np.concatenate(out, axis=0)


def _recent_store_hits(tokens, score):
    hits = np.zeros_like(score)
    for r, (t, s) in enumerate(zip(tokens, score)):
        last = None
        for i in range(len(t) - 1):
            if s[i]:
                hits[r, i] = last == t[i + 1]
            if t[i] == STORE and i + 2 < len(t):
                last = int(t[i + 2])
    return hits


def by_bucket(hits: np.ndarray, delays: np.ndarray, score: np.ndarray) -> dict:
    out = {}
    for lo, hi in BUCKETS:
        m = score & (delays >= lo) & (delays < hi)
        if m.any():
            out[bucket_name(lo, hi)] = {"accuracy": float(hits[m].mean()), "n": int(m.sum())}
    return out


def test_sets(n_test: int) -> dict:
    sets = {}
    for t in TASKS:
        d = dataset(t, "test", n_test if SPECS[t].length <= 256 else n_test // 2, DATA_CACHE)
        d["delay"] = np.stack([delay_of(tok, sc) for tok, sc in zip(d["tokens"], d["score"])])
        sets[t] = d
    return sets


def baseline(n_test: int) -> dict:
    return {t: by_bucket(_recent_store_hits(d["tokens"], d["score"]), d["delay"], d["score"]) for t, d in test_sets(n_test).items()}


def run(device: str = "cuda", log=print) -> int:
    done, cache = 0, {}
    for rpath in sorted((OUT / "runs").glob("*/result.json")):
        res = json.loads(rpath.read_text(encoding="utf-8"))
        mpath, dpath = rpath.with_name("model.pt"), rpath.with_name("delay.json")
        if not any(t.startswith("kvd") for t in res["tasks"]) or not mpath.exists():
            continue
        if dpath.exists() and json.loads(dpath.read_text(encoding="utf-8")).get("version") == DELAY_VERSION:
            continue
        cfg = res["config_key"]
        sets = cache.setdefault(cfg["n_test"], test_sets(cfg["n_test"]))
        model = build(res["variant"], VOCAB).to(device)
        model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
        out = {t: by_bucket(hits_per_position(model, d["tokens"], cfg["memory"], cfg["chunk"], torch.device(device)), d["delay"], d["score"])
               for t, d in sets.items()}
        dpath.write_text(json.dumps({"version": DELAY_VERSION, "test": out}, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"  {res['run_id']}: " + " ".join(f"{k} {v['accuracy']:.2f}" for t in TASKS for k, v in out[t].items()))
        done += 1
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    print(f"지연 구간별 평가 {run(ap.parse_args().device)}개")
