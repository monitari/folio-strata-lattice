"""새 과제 묶음(계획 H)의 난이도 손잡이별 정확도 → runs/<id>/knobs.json (재학습 없음, 저장된 모델로 평가)

과제마다 생성기가 남긴 위치별 난이도(meta["knob"]: 걸음 수 k · 나머지 m · 원문 길이 n · 깊이 · 주기 P)로
채점 위치를 나눠 정확도를 낸다. 같은 구간으로 단순 규칙 기준선도 낸다. 평균 하나로는 '어디서 무너지는지'가 안 보이기 때문이다.

    py -m candidates.knobs
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .benchmark import DATA_CACHE, OUT
from .delay import hits_per_position
from .models import build
from .tasks import (ARROW, CLOSE_A, CLOSE_B, MARK_COPY, MARK_REV, OPEN_A, OPEN_B, SEP, SPECS, SUITE, SUITE_TESTS, VOCAB, dataset,
                    episode)

KNOB_VERSION = "knob-v1"


def test_sets(n_test: int) -> dict:
    sets = {}
    for t in SUITE_TESTS:
        n = n_test if SPECS[t].length <= 256 else n_test // 2
        d = dict(dataset(t, "test", n, DATA_CACHE))
        d["knob"] = np.stack([episode(t, "test", i).meta["knob"] for i in range(n)])
        sets[t] = d
    return sets


def by_knob(hits: np.ndarray, knob: np.ndarray, score: np.ndarray) -> dict:
    """{손잡이 값: {accuracy, n}} + 전체 채점 위치의 정확도 'all'."""
    out = {"all": {"accuracy": float(hits[score].mean()), "n": int(score.sum())}}
    for v in np.unique(knob[knob >= 0]):
        m = knob == v
        out[str(int(v))] = {"accuracy": float(hits[m].mean()), "n": int(m.sum())}
    return out


# ---------------------------------------------------------------- 단순 규칙 (위치별 정답 여부)
def _rule_one_step(tok, sc):
    """순환 사슬: k 를 무시하고 한 걸음 답을 낸다 (k = 1 에서만 맞음)."""
    hits = np.zeros_like(sc)
    for r, (t, s) in enumerate(zip(tok, sc)):
        succ = {}
        for i in range(len(t) - 1):
            if t[i] == SEP:
                succ = {}
            elif i + 2 < len(t) and t[i + 1] == ARROW:
                succ[int(t[i])] = int(t[i + 2])
            if s[i]:
                hits[r, i] = succ.get(int(t[i - 1])) == t[i + 1]
    return hits


def _rule_last_answer(tok, sc):
    """셈 나머지: 같은 m 의 직전 답을 그대로 낸다 (그 사이 셀 기호가 m의 배수만큼 나왔을 때만 맞음)."""
    hits = np.zeros_like(sc)
    for r, (t, s) in enumerate(zip(tok, sc)):
        last = {}
        for i in range(len(t) - 1):
            if s[i]:
                m = int(t[i])
                hits[r, i] = last.get(m) == t[i + 1]
                last[m] = int(t[i + 1])
    return hits


def _rule_always_copy(tok, sc):
    """베끼기·뒤집기: 표시를 무시하고 늘 그대로 베낀다 (뒤집기 문제는 틀림)."""
    hits = np.zeros_like(sc)
    for r, (t, s) in enumerate(zip(tok, sc)):
        i = 0
        while i < len(t) and t[i] == SEP:
            j = i + 1
            while t[j] not in (MARK_COPY, MARK_REV):
                j += 1
            x = t[i + 1:j]
            n = len(x)
            hits[r, j:j + n] = t[j + 1:j + 1 + n] == x
            i = j + 1 + n
    return hits


def _rule_last_open(tok, sc):
    """괄호 깊이: 닫기를 무시하고 기록에서 가장 최근에 연 괄호를 닫는다 (바로 앞에서 열었을 때만 맞음)."""
    hits = np.zeros_like(sc)
    for r, (t, s) in enumerate(zip(tok, sc)):
        last = None
        for i in range(len(t) - 1):
            if t[i] in (OPEN_A, OPEN_B):
                last = int(t[i])
            if s[i]:
                hits[r, i] = (CLOSE_A if last == OPEN_A else CLOSE_B) == t[i + 1]
    return hits


def _rule_fast_only(tok, sc):
    """느린 주기: 빠른 무늬만 안다 (3칸 전 기호를 낸다). 오프셋이 바뀌는 자리는 틀린다."""
    hits = np.zeros_like(sc)
    hits[:, 2:-1] = tok[:, 3:] == tok[:, :-3]
    return hits & sc


RULES = {"cyc": ("한 걸음 답", _rule_one_step), "count": ("같은 m 직전 답", _rule_last_answer),
         "copy": ("늘 그대로 베끼기", _rule_always_copy), "paren": ("가장 최근에 연 괄호", _rule_last_open),
         "slowp": ("빠른 무늬만 (3칸 전)", _rule_fast_only)}


def rule_for(task: str):
    return RULES[task.split("_")[0]]


def baseline(n_test: int) -> dict:
    out = {}
    for t, d in test_sets(n_test).items():
        name, fn = rule_for(t)
        out[t] = {"name": name, **by_knob(fn(d["tokens"], d["score"]), d["knob"], d["score"])}
    return out


def run(device: str = "cuda", log=print) -> int:
    done, cache = 0, {}
    for rpath in sorted((OUT / "runs").glob("*/result.json")):
        res = json.loads(rpath.read_text(encoding="utf-8"))
        mpath, kpath = rpath.with_name("model.pt"), rpath.with_name("knobs.json")
        if not set(res["tasks"]) <= set(SUITE) or not mpath.exists():
            continue
        if kpath.exists() and json.loads(kpath.read_text(encoding="utf-8")).get("version") == KNOB_VERSION:
            continue
        cfg = res["config_key"]
        sets = cache.setdefault(cfg["n_test"], test_sets(cfg["n_test"]))
        model = build(res["variant"], VOCAB).to(device)
        model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
        out = {t: by_knob(hits_per_position(model, d["tokens"], cfg["memory"], cfg["chunk"], torch.device(device)), d["knob"], d["score"])
               for t, d in sets.items()}
        kpath.write_text(json.dumps({"version": KNOB_VERSION, "test": out}, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"  {res['run_id']}: " + " ".join(f"{t} {v['all']['accuracy']:.2f}" for t, v in out.items()))
        done += 1
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    print(f"손잡이별 평가 {run(ap.parse_args().device)}개")
