"""학습이 끝난 모델을 추가 시험으로만 평가한다 (재학습 없음) → runs/<id>/posthoc.json

왜 따로 두나
  기본 과제(tasks.CORE)를 바꾸면 data_fingerprint 가 바뀌어 모든 실행의 판별 키가 달라진다 (= 전부 재학습).
  그래서 학습이 끝난 뒤 발견한 문제를 보완하는 시험은 여기에 둔다.

관계 사슬 단계 수 일반화 (chain_hop3/4/5)
  tasks.py 의 chain_long (사슬 1개, 첫 개체 질의) 은 답이 늘 '화살표 오른쪽에만 나오는 유일한 개체' 라서
  사슬을 따라가지 않고 끝 개체만 찾아도 풀린다. 단계 수 일반화의 증거로 쓸 수 없다.
  여기서는 개체 8개 안에서 사슬을 늘 2개 두고, 두 사슬의 머리를 모두 질의한다 (순서 무작위).
  답 후보(끝 개체)가 둘이므로 u 가 어느 사슬에 속하는지 따라가야 한다.
    chain_hop3 : 사슬 3+3 단계, 두 질의 모두 채점 (학습 분포의 가장 긴 경우)
    chain_hop4 : 사슬 4+2 단계, 4단계 질의만 채점 (학습보다 1단계 김)
    chain_hop5 : 사슬 5+1 단계, 5단계 질의만 채점 (학습보다 2단계 김)
  남은 지름길: 짧은 사슬의 구성원인지 확인하면 긴 사슬 질의의 답을 소거법으로 고를 수 있다.
  hop4 에서는 짧은 사슬이 2단계여서 그 확인에도 2단계가 필요하지만, hop5 의 짧은 사슬은 사실 1개뿐이다.

    py -m candidates.posthoc            # 끝난 실행 모두 (관계 사슬로 학습한 것만)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .benchmark import DATA_CACHE, OUT, evaluate
from .models import build
from .tasks import ARROW, ENT, N_SYM, QUERY, SEP, VOCAB, Episode, _empty

POSTHOC_VERSION = "posthoc-v1"
SEED_BASE = 40_000_000                     # 학습·검증·시험(10M/20M/30M)과 겹치지 않는 난수
HOPS = {"chain_hop3": ((3, 3), (0, 1)), "chain_hop4": ((4, 2), (0,)), "chain_hop5": ((5, 1), (0,))}
NAMES = {"chain_hop3": "3단계 (학습 범위)", "chain_hop4": "4단계 (+1)", "chain_hop5": "5단계 (+2)"}
LENGTH, N_EPISODES = 256, 128


def chain_hops(edges: tuple[int, int], scored: tuple[int, ...], L: int, rng: np.random.Generator) -> Episode:
    tokens, score, oracle = _empty(L)
    i = 0
    while True:
        ents = rng.permutation(N_SYM) + ENT
        nxt, facts, chains, used = {}, [], [], 0
        for n in edges:
            nodes = [int(x) for x in ents[used:used + n + 1]]
            used += n + 1
            chains.append(nodes)
            for a, b in zip(nodes, nodes[1:]):
                nxt[a] = b
                facts.append((a, b))
        rng.shuffle(facts)
        toks, sc, orc = [SEP], [False], [-1]
        for a, b in facts:
            toks += [a, ARROW, b]
            sc += [False] * 3
            orc += [-1] * 3
        for c in rng.permutation(len(chains)):
            head, end = chains[c][0], chains[c][-1]
            toks += [QUERY, head, end]
            sc += [False, int(c) in scored, False]
            orc += [-1, end, -1]
        if i + len(toks) > L:
            break
        tokens[i:i + len(toks)], score[i:i + len(toks)], oracle[i:i + len(toks)] = toks, sc, orc
        i += len(toks)
    return Episode(tokens, score, oracle, {"queries": int(score.sum())})


def check_hops(tokens: np.ndarray, score: np.ndarray, hops: int) -> bool:
    """채점 위치마다 앞선 사실로 답을 다시 계산하고, 경로 길이가 정확히 hops 인지 확인."""
    nxt, i = {}, 0
    for t in np.flatnonzero(score):
        while i < t - 1:                       # t-1 = QUERY, t = 질의 개체, t+1 = 답
            if tokens[i] == SEP:
                nxt = {}
            elif tokens[i + 1] == ARROW and tokens[i] != QUERY:
                nxt[int(tokens[i])] = int(tokens[i + 2])
                i += 3
                continue
            i += 1
        if tokens[t - 1] != QUERY:
            return False
        end, n = int(tokens[t]), 0
        while end in nxt:
            end, n = nxt[end], n + 1
        if end != tokens[t + 1] or n != hops:
            return False
    return bool(score.any())


def dataset(task: str) -> dict[str, np.ndarray]:
    path = DATA_CACHE / f"{task}_test_{N_EPISODES}_{POSTHOC_VERSION}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    edges, scored = HOPS[task]
    k = list(HOPS).index(task)
    eps = [chain_hops(edges, scored, LENGTH, np.random.default_rng(SEED_BASE + k * 100_000 + j)) for j in range(N_EPISODES)]
    data = {"tokens": np.stack([e.tokens for e in eps]), "score": np.stack([e.score for e in eps]),
            "oracle": np.stack([e.oracle for e in eps])}
    hops = edges[0]
    assert all(check_hops(e.tokens, e.score, hops) for e in eps), task
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def run(device: str = "cuda", log=print) -> int:
    datas = {t: dataset(t) for t in HOPS}
    done = 0
    for run_dir in sorted((OUT / "runs").glob("*")):
        rpath, mpath, ppath = run_dir / "result.json", run_dir / "model.pt", run_dir / "posthoc.json"
        if not (rpath.exists() and mpath.exists()):
            continue
        res = json.loads(rpath.read_text(encoding="utf-8"))
        if "chain" not in res["tasks"]:
            continue
        if ppath.exists() and json.loads(ppath.read_text(encoding="utf-8")).get("version") == POSTHOC_VERSION:
            continue
        model = build(res["variant"], VOCAB).to(device)
        model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
        cfg = res["config_key"]
        with torch.no_grad():
            out = {t: evaluate(model, d, cfg["memory"], cfg["chunk"], torch.device(device)) for t, d in datas.items()}
        ppath.write_text(json.dumps({"version": POSTHOC_VERSION, "test": out}, indent=1), encoding="utf-8")
        log(f"  {res['run_id']}: " + " ".join(f"{t} {v['accuracy']:.3f}" for t, v in out.items()))
        done += 1
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    print(f"추가 평가 {run(ap.parse_args().device)}개")
