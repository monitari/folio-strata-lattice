"""웹 화면(web/)에 줄 자료를 한 JSON 으로 모은다. server.py 의 /api/results 가 이것을 돌려준다.

실행별 결과는 요약하지 않고 그대로 보낸다 (평균·편차·그림은 화면에서 계산). 여기서 더 계산하는 것은
과제별 '단순 규칙' 기준선뿐이다: 모델이 정말 배운 것인지, 쉬운 규칙만 따라 한 것인지 가르기 위해서다.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import delay, knobs, posthoc
from .benchmark import DATA_CACHE, OUT
from .models import VARIANTS, build, describe, n_params
from .plan import ABLATIONS, GROUPS, IMPROVED, MAIN, configs
from .tasks import (ARROW, CORE, DATA_VERSION, EXTRA_VERSION, N_SYM, QUERY, SEP, SPECS, STORE, SUITE_TESTS, SUITE_VERSION, VOCAB,
                    dataset)

BASELINE_PATH = OUT / "baselines.json"
BASELINE_KEY = f"{DATA_VERSION}|{EXTRA_VERSION}|{SUITE_VERSION}|{posthoc.POSTHOC_VERSION}|{','.join(SPECS)}"
# 새 과제의 우연 수준: 순환 사슬 개체 8 · 셈 나머지 m 평균 · 베끼기 기호 16 · 괄호 2종 · 느린 주기 요리 8
CHANCE = {"cyc": 1 / 8, "count": (1 / 2 + 1 / 3 + 1 / 5) / 3, "copy": 1 / 16, "paren": 1 / 2, "slowp": 1 / 8}
N_TEST = 128                                    # 계획의 n_test (길이 1024 과제는 절반)


# ---------------------------------------------------------------- 단순 규칙 기준선
def _recent_store(tokens, score) -> float:
    """키-값: 키를 보지 않고 '가장 최근에 저장된 값'을 답한다."""
    hit = n = 0
    for t, s in zip(tokens, score):
        last = None
        for i in range(len(t) - 1):
            if s[i]:
                n += 1
                hit += last == t[i + 1]
            if t[i] == STORE and i + 2 < len(t):
                last = int(t[i + 2])
    return hit / max(n, 1)


def _tail_guess(tokens, score) -> float:
    """관계 사슬: 사슬을 따라가지 않고, 화살표 오른쪽에만 나온 개체(끝 개체) 중 하나를 고른다 (기댓값)."""
    hit = n = 0
    for t, s in zip(tokens, score):
        left, right = set(), set()
        for i in range(len(t) - 1):
            if t[i] == SEP:
                left, right = set(), set()
            elif i + 2 < len(t) and t[i + 1] == ARROW and t[i] != QUERY:
                left.add(int(t[i]))
                right.add(int(t[i + 2]))
            if s[i]:
                tails = right - left
                n += 1
                hit += (1 / len(tails)) if int(t[i + 1]) in tails else 0
    return hit / max(n, 1)


def baselines() -> dict:
    if BASELINE_PATH.exists():
        saved = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if saved.get("key") == BASELINE_KEY:
            return saved["rules"]
    rules = {}
    for key, spec in SPECS.items():
        d = dataset(key, "test", N_TEST if spec.length <= 256 else N_TEST // 2, DATA_CACHE)
        if key.startswith("kv"):
            rules[key] = {"name": "가장 최근 저장값", "accuracy": float(_recent_store(d["tokens"], d["score"]))}
        elif key.startswith("chain"):
            rules[key] = {"name": "끝 개체 찍기", "accuracy": _tail_guess(d["tokens"], d["score"])}
        elif key in SUITE_TESTS:
            name, fn = knobs.rule_for(key)
            rules[key] = {"name": name, "accuracy": float(fn(d["tokens"], d["score"])[d["score"]].mean())}
    for key in posthoc.HOPS:
        d = posthoc.dataset(key)
        rules[key] = {"name": "끝 개체 찍기", "accuracy": _tail_guess(d["tokens"], d["score"])}
    BASELINE_PATH.write_text(json.dumps({"key": BASELINE_KEY, "rules": rules}, ensure_ascii=False, indent=1), encoding="utf-8")
    return rules


# ---------------------------------------------------------------- 묶기
def _runs() -> list[dict]:
    out = []
    for p in sorted((OUT / "runs").glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        ph = p.with_name("posthoc.json")
        out.append({
            "id": r["run_id"], "variant": r["variant"], "mode": r["mode"], "tasks": r["tasks"], "memory": r["memory"],
            "seed": r["seed"], "steps": r["config_key"]["steps"], "chunk": r["config_key"]["chunk"], "train_tokens": r["train_tokens"],
            "test": {t: {"accuracy": v["accuracy"], "bits": v["bits"], "oracle": v["oracle_accuracy"]} for t, v in r["test"].items()},
            "history": [{"step": h["step"], "val": {t: v["accuracy"] for t, v in h["val"].items()}} for h in r["history"]],
            "posthoc": ({t: v["accuracy"] for t, v in json.loads(ph.read_text(encoding="utf-8"))["test"].items()}
                        if ph.exists() else None),
            "delay": (json.loads(p.with_name("delay.json").read_text(encoding="utf-8"))["test"]
                      if p.with_name("delay.json").exists() else None),
            "knobs": (json.loads(p.with_name("knobs.json").read_text(encoding="utf-8"))["test"]
                      if p.with_name("knobs.json").exists() else None),
        })
    return out


def plan_status() -> dict:
    status = {}
    for g, label in GROUPS.items():
        cs = configs(g)
        done = sum((OUT / "runs" / c.run_id() / "result.json").exists() for c in cs)
        status[g] = {"label": label, "done": done, "total": len(cs)}
    return status


def payload() -> dict:
    rules = baselines()
    runs = _runs()
    tasks = [{"key": s.key, "name": s.name, "kind": s.kind, "length": s.length, "skill": s.skill, "train": s.train, "core": s.key in CORE,
              "suite": s.key in SUITE_TESTS, "chance": CHANCE.get(s.key.split("_")[0], 1 / N_SYM), "rule": rules.get(s.key)}
             for s in SPECS.values()]
    tasks += [{"key": k, "name": f"관계 사슬 {n}", "kind": "query", "length": posthoc.LENGTH, "skill": "사슬 단계 수",
               "train": False, "posthoc": True, "chance": 1 / N_SYM, "rule": rules.get(k)} for k, n in posthoc.NAMES.items()]
    models = {}
    for v in VARIANTS:
        m = build(v, VOCAB)
        models[v] = {"params": n_params(m), "config": describe(m)["config"]}
    read = lambda name: json.loads((OUT / name).read_text(encoding="utf-8")) if (OUT / name).exists() else None
    delay_base = delay.baseline(256) if any(r["delay"] for r in runs) else None
    knob_base = knobs.baseline(N_TEST) if any(r["knobs"] for r in runs) else None
    return {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "main": MAIN, "ablations": ABLATIONS, "improved": IMPROVED,
            "delay_baseline": delay_base, "knob_baseline": knob_base,
            "tasks": tasks, "models": models, "runs": runs, "plan": plan_status(),
            "measure": read("measure.json"), "checks": read("checks.json")}
