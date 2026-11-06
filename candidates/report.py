"""예비 비교 결과를 표로 정리한다 → results/candidates/report.md, summary.json

- 과제별 정확도·손실: 초기화 난수별 값과 평균·표준편차·범위
- 조건 구분: 공동/과제별 학습 × 지속/창 기억
- 학습 곡선 판정: 마지막 두 검증 사이에도 오르고 있으면 '아직 오르는 중' (덜 학습됨과 구조의 한계를 가르기 위한 표시)
- 자원: measure.json (단독 실행에서 잰 값만)
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from .benchmark import OUT
from .tasks import SPECS, SUITE

MAIN = ["transformer", "gru", "folio", "strata", "lattice"]
ABLATIONS = {"folio": ["folio_nomem", "folio_r1"], "strata": ["strata_sameperiod", "strata_noaux"],
             "lattice": ["lattice_k1", "lattice_nonbr"]}
NAMES = {"transformer": "Transformer", "gru": "GRU", "folio": "Folio", "strata": "Strata", "lattice": "Lattice",
         "folio_nomem": "Folio 기억 제거", "folio_r1": "Folio 읽기 1회", "strata_sameperiod": "Strata 모든 수준 매 입력",
         "strata_noaux": "Strata 보조 손실 제거", "lattice_k1": "Lattice 반복 1회", "lattice_nonbr": "Lattice 이웃 메시지 제거",
         "folio2": "Folio2 (개선판)", "lattice2": "Lattice2 (개선판)", "transformer256": "Transformer 창 256"}
IMPROVED = ["folio2", "lattice2"]
SUITE_ROSTER = ["transformer", "transformer256", "gru", "folio", "folio2", "strata", "strata_sameperiod", "lattice", "lattice2"]
# 새 과제 묶음의 난이도 손잡이: 표 제목, (시험 과제, 손잡이 값) 목록
KNOB_TABLES = [
    ("순환 사슬: 걸음 수 k (4 이상은 학습보다 많음)", [("cyc", k) for k in (1, 2, 3)] + [("cyc_far", k) for k in (4, 5, 6, 7)]),
    ("셈 나머지: m (왼쪽 256기호, 오른쪽 1024기호)", [("count", m) for m in (2, 3, 5)] + [("count_long", m) for m in (2, 3, 5)]),
    ("베끼기·뒤집기: 원문 길이 n (28 이상은 학습보다 김)", [("copy", n) for n in (4, 12, 24)] + [("copy_long", n) for n in (28, 36, 48)]),
    ("괄호 깊이: 연달아 닫기 전 깊이 (4 이상은 학습보다 깊음)", [("paren", d) for d in (1, 2, 3)] + [("paren_deep", d) for d in (4, 5, 6, 7)]),
    ("느린 주기: 오프셋이 바뀌는 자리, 주기 P (256 은 학습보다 김)", [("slowp", P) for P in (32, 64, 128)] + [("slowp_long", 256)]),
]
STILL_RISING = 0.01      # 마지막 검증 간격의 평균 정확도 상승이 이보다 크면 '아직 오르는 중'
STEPS = {"joint": 800, "separate": 400}   # 예비 비교의 학습 길이. 그 밖의 길이는 '학습 길이' 표로만 보인다
NOTES = {"chain_long": "관계 사슬 (긴 사슬)* "}


def load_runs() -> list[dict]:
    runs = []
    for p in sorted((OUT / "runs").glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        ph = p.with_name("posthoc.json")
        if ph.exists():
            r["posthoc"] = json.loads(ph.read_text(encoding="utf-8"))["test"]
        dl = p.with_name("delay.json")
        if dl.exists():
            r["delay"] = json.loads(dl.read_text(encoding="utf-8"))["test"]
        kn = p.with_name("knobs.json")
        if kn.exists():
            r["knobs"] = json.loads(kn.read_text(encoding="utf-8"))["test"]
        runs.append(r)
    return runs


def _stat(vals):
    if not vals:
        return None
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "n": len(vals), "values": vals}


def group(runs):
    """(variant, mode, tasks, memory, steps) → 난수별 실행 목록."""
    g = defaultdict(list)
    for r in runs:
        g[(r["variant"], r["mode"], tuple(r["tasks"]), r["memory"], r["config_key"]["steps"], r["config_key"]["chunk"])].append(r)
    return g


def summarize(rs: list[dict]) -> dict:
    tasks = rs[0]["test"].keys()
    out = {"seeds": [r["seed"] for r in rs], "params": rs[0]["params"], "train_tokens": rs[0]["train_tokens"],
           "test": {t: {m: _stat([r["test"][t][m] for r in rs]) for m in ("accuracy", "bits")} for t in tasks},
           "oracle": {t: rs[0]["test"][t]["oracle_accuracy"] for t in tasks}}
    if all("posthoc" in r for r in rs):
        out["posthoc"] = {t: _stat([r["posthoc"][t]["accuracy"] for r in rs]) for t in rs[0]["posthoc"]}
    # 학습 곡선: 과제 평균 검증 정확도의 마지막 상승폭
    rises = []
    for r in rs:
        h = r["history"]
        if len(h) >= 2:
            a = statistics.fmean(v["accuracy"] for v in h[-1]["val"].values())
            b = statistics.fmean(v["accuracy"] for v in h[-2]["val"].values())
            rises.append(a - b)
    out["last_rise"] = statistics.fmean(rises) if rises else None
    out["curve"] = [{"step": h["step"], "val_acc": statistics.fmean(statistics.fmean(v["accuracy"] for v in hh["val"].values())
                                                                    for hh in [r["history"][i] for r in rs])}
                    for i, h in enumerate(rs[0]["history"])]
    return out


def _fmt(s, pct=True):
    if s is None:
        return "–"
    if pct:
        return f"{s['mean'] * 100:.1f} ± {s['sd'] * 100:.1f}"
    return f"{s['mean']:.2f} ± {s['sd']:.2f}"


def table(summary: dict, variants: list[str], key_filter, title: str, metric: str = "accuracy") -> list[str]:
    rows = [(v, summary[k]) for v in variants for k in summary if k[0] == v and key_filter(k)]
    if not rows:
        return []
    tasks = list(rows[0][1]["test"])
    head = "| 모델 | 파라미터 | " + " | ".join(NOTES.get(t, SPECS[t].name) for t in tasks) + " | 마지막 상승 |"
    lines = [f"\n### {title}\n", head, "|" + "---|" * (len(tasks) + 3)]
    for v, s in rows:
        rise = s["last_rise"]
        flag = "–" if rise is None else (f"{rise * 100:+.1f}%p ⚠" if rise > STILL_RISING else f"{rise * 100:+.1f}%p")
        lines.append(f"| {NAMES.get(v, v)} | {s['params']:,} | " + " | ".join(
            _fmt(s["test"][t][metric], metric == "accuracy") for t in tasks) + f" | {flag} |")
    orc = rows[0][1]["oracle"]
    if metric == "accuracy":
        lines.append("| 이론적 최대 | | " + " | ".join(f"{orc[t] * 100:.1f}" for t in tasks) + " | |")
    return lines


def posthoc_table(summary: dict) -> list[str]:
    from .posthoc import NAMES as HOP_NAMES
    rows = [(k, s) for k, s in summary.items() if "posthoc" in s and k[4] == STEPS[k[1]]
            and (k[0] in MAIN or k[1] == "joint")]
    if not rows:
        return []
    order = {v: i for i, v in enumerate(MAIN + [a for b in ABLATIONS.values() for a in b])}
    rows.sort(key=lambda r: (r[0][1], r[0][3], order.get(r[0][0], 99)))
    tasks = list(rows[0][1]["posthoc"])
    lines = ["\n### 관계 사슬 단계 수 일반화 (추가 시험, candidates/posthoc.py)\n",
             "사슬 2개 · 두 머리를 모두 질의. 학습은 1~3단계. 재학습 없이 저장된 모델로 평가 (난수별 평균 ± 표준편차, %). "
             "기준선: 끝 개체(화살표 오른쪽에만 나오는 개체) 둘 중 하나를 고르면 50%. "
             "5단계는 짧은 사슬이 사실 1개뿐이라 소거법이 쉬워 50% 를 넘기 쉽다.", "",
             "| 학습 | 기억 | 모델 | " + " | ".join(HOP_NAMES[t] for t in tasks) + " |", "|---|---|---|" + "---|" * len(tasks)]
    for k, s in rows:
        mode = "공동" if k[1] == "joint" else "관계 사슬만"
        lines.append(f"| {mode} | {'지속' if k[3] == 'persistent' else '창'} | {NAMES.get(k[0], k[0])} | "
                     + " | ".join(_fmt(s["posthoc"][t]) for t in tasks) + " |")
    return lines


def length_table(summary: dict) -> list[str]:
    """같은 모델·조건을 더 길게 학습한 실행이 있으면 예비 길이(난수 0)와 나란히 보인다."""
    longer = {k: s for k, s in summary.items() if k[4] != STEPS[k[1]] and k[0] in MAIN and not _is_kvd(k) and k[2] != SUITE}
    if not longer:
        return []
    lines = []
    for k, s in sorted(longer.items(), key=lambda kv: (kv[0][4], MAIN.index(kv[0][0]))):
        base = summary.get(k[:4] + (STEPS[k[1]], k[5]))
        if not lines:
            tasks = list(s["test"])
            lines = [f"\n### 학습 길이: {STEPS[k[1]]}걸음 → {k[4]}걸음 (공동 학습 · 지속 기억 · 정확도 %, 난수 0)\n",
                     "덜 학습된 것인지 구조의 한계인지 가르기 위한 진단. 한 난수만이라 차이가 작으면 해석하지 않는다.", "",
                     "| 모델 | 걸음 | " + " | ".join(NOTES.get(t, SPECS[t].name) for t in tasks) + " | 마지막 상승 |",
                     "|---|---|" + "---|" * (len(tasks) + 1)]
        for steps, ss in ((STEPS[k[1]], base), (k[4], s)):
            if ss is None:
                continue
            i = ss["seeds"].index(0) if 0 in ss["seeds"] else None
            if i is None:
                continue
            rise = ss["last_rise"] if steps == k[4] else None
            lines.append(f"| {NAMES[k[0]]} | {steps} | " + " | ".join(f"{ss['test'][t]['accuracy']['values'][i] * 100:.1f}" for t in tasks)
                         + f" | {'–' if rise is None else f'{rise * 100:+.1f}%p'} |")
    return lines


def _is_kvd(k) -> bool:
    return any(t.startswith("kvd") for t in k[2])


def delay_table(g) -> list[str]:
    """지연 사다리 키-값: 지연 구간별 정확도 (candidates/delay.py). 청크 길이별로 따로."""
    from .delay import BUCKETS, TASKS, bucket_name
    rows = [(k, rs) for k, rs in g.items() if _is_kvd(k) and all(r.get("delay") for r in rs)]
    if not rows:
        return []
    names = [bucket_name(*b) for b in BUCKETS]
    lines = ["\n### 키-값 지연 사다리: 지연 구간별 정확도 (%, 난수별 평균 ± 표준편차)\n",
             "학습 지연 6–192기호(에피소드 256), 시험은 숨긴 키 질의만 채점한다. 256기호까지는 같은 분포의 시험(256), 그 뒤는 240–900기호 시험(1024). "
             "청크 256 은 에피소드 전체에 기울기가 닿는다. 학습 채점 '숨긴 키' = kvd, '모든 질의' = kvd_all.", "",
             "| 학습 채점 | 모델 | 청크 | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for k, rs in sorted(rows, key=lambda kv: (kv[0][2], MAIN.index(kv[0][0]) if kv[0][0] in MAIN else 9, -kv[0][5])):
        cells = []
        for i, b in enumerate(names):
            t = TASKS[0] if BUCKETS[i][1] <= 256 else TASKS[1]
            vals = [r["delay"][t][b]["accuracy"] for r in rs if b in r["delay"][t]]
            cells.append(_fmt(_stat(vals)) if vals else "–")
        mode = "모든 질의" if "kvd_all" in k[2] else "숨긴 키"
        lines.append(f"| {mode} | {NAMES.get(k[0], k[0])} | {k[5]} | " + " | ".join(cells) + " |")
    return lines


def knob_tables(g) -> list[str]:
    """새 과제 묶음: 난이도 손잡이별 정확도 (candidates/knobs.py)."""
    rows = [(k, rs) for k, rs in g.items() if k[2] == SUITE and all(r.get("knobs") for r in rs)]
    if not rows:
        return []
    rows.sort(key=lambda kv: SUITE_ROSTER.index(kv[0][0]) if kv[0][0] in SUITE_ROSTER else 99)
    lines = ["\n## 새 과제 묶음 (계획 H): 난이도 손잡이별 정확도 (%, 난수별 평균 ± 표준편차)\n",
             "다섯 과제 공동 학습 · 3000걸음. 순환형은 청크 256, Transformer 는 창 64(청크 64) · 창 256. 재학습 없이 저장된 모델로 평가."]
    for title, cols in KNOB_TABLES:
        lines += [f"\n### {title}\n", "| 모델 | " + " | ".join(f"{t.split('_')[1] + ' ' if '_' in t else ''}{v}" for t, v in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for k, rs in rows:
            cells = []
            for t, v in cols:
                vals = [r["knobs"][t][str(v)]["accuracy"] for r in rs if str(v) in r["knobs"].get(t, {})]
                cells.append(_fmt(_stat(vals)) if vals else "–")
            lines.append(f"| {NAMES.get(k[0], k[0])} | " + " | ".join(cells) + " |")
    return lines


def write_report() -> Path:
    runs = load_runs()
    g = group(runs)
    summary = {k: summarize(v) for k, v in g.items()}
    pre = lambda k: k[4] == STEPS[k[1]]
    lines = ["# 후보 모델 예비 비교 결과", "",
             f"실행 {len(runs)}개. 각 칸은 초기화 난수별 시험 정확도(%)의 평균 ± 표준편차. "
             "'마지막 상승'은 마지막 두 검증 사이의 과제 평균 검증 정확도 변화이며, ⚠ 는 아직 오르는 중(덜 학습됨 가능성)이다. "
             "상승이 작아도 정체 구간일 수 있으므로 ⚠ 가 없다고 다 배운 것은 아니다 (학습 길이 표 참고).", "",
             "\\* 관계 사슬 (긴 사슬) 은 사슬 1개 · 첫 개체 질의라서 답이 늘 '화살표 오른쪽에만 나오는 개체' 다. "
             "사슬을 따라가지 않고 끝 개체만 찾아도 풀리므로 단계 수 일반화의 증거가 아니다. 그 대신 아래 '단계 수 일반화' 표를 본다.", ""]
    for memory in ("persistent", "window"):
        lines += table(summary, MAIN, lambda k, m=memory: pre(k) and k[1] == "joint" and k[3] == m,
                       f"공동 학습 · {'지속 기억' if memory == 'persistent' else '창 초기화'} · 정확도")
        lines += table(summary, MAIN, lambda k, m=memory: pre(k) and k[1] == "joint" and k[3] == m,
                       f"공동 학습 · {'지속 기억' if memory == 'persistent' else '창 초기화'} · 손실 (bits)", "bits")
    for task in ("kv", "mixed", "chain"):
        lines += table(summary, MAIN, lambda k, t=task: pre(k) and k[1] == "separate" and k[2] == (t,) and k[3] == "persistent",
                       f"과제별 학습 ({SPECS[task].name}) · 지속 기억 · 정확도")
    for base, abls in ABLATIONS.items():
        lines += table(summary, [base] + abls, lambda k: pre(k) and k[1] == "joint" and k[3] == "persistent",
                       f"대조군: {NAMES[base]} · 공동 학습 · 지속 기억 · 정확도")
    lines += table(summary, MAIN + IMPROVED, lambda k: k[1] == "joint" and k[3] == "persistent" and k[4] == 3000 and k[2] != SUITE,
                   "공동 학습 · 지속 기억 · 3000걸음 · 개선판 포함 · 정확도")
    lines += table(summary, SUITE_ROSTER, lambda k: k[2] == SUITE, "새 과제 묶음 · 공동 학습 · 3000걸음 · 정확도")
    lines += posthoc_table(summary)
    lines += length_table(summary)
    lines += delay_table(g)
    lines += knob_tables(g)

    lines += ["\n### 초기화 난수별 값 (공동 학습 · 지속 기억 · 정확도 %)\n", "| 모델 | 과제 | " + " | ".join(f"난수 {i}" for i in range(3)) + " |", "|---|---|---|---|---|"]
    for k, s in summary.items():
        if k[0] in MAIN and k[1] == "joint" and k[3] == "persistent" and pre(k):
            for t, m in s["test"].items():
                lines.append(f"| {NAMES[k[0]]} | {SPECS[t].name} | " + " | ".join(f"{v * 100:.1f}" for v in m["accuracy"]["values"]) + " |")

    mpath = OUT / "measure.json"
    if mpath.exists():
        meas = json.loads(mpath.read_text(encoding="utf-8"))
        lines += ["\n### 자원 (단독 실행에서 측정, 동기화·준비 실행 포함)\n",
                  f"측정 시각 {meas['measured_at']}. 학습: 배치 {meas['rows'][0]['setup']['batch']} × 256 토큰, 청크 64 (Transformer 창 256 은 청크 256), 지속 기억 경로. "
                  "지연: 배치 1 에서 step 한 번 (Transformer 는 창을 채운 상태, KV 캐시 없이 창 전체 재계산).", "",
                  "| 모델 | 파라미터 | 학습 ms/걸음 | 학습 토큰/초 | 학습 피크 메모리 | 지연 GPU (평균/p90) | 지연 CPU (평균/p90) | 지속 상태 (배치 1) |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in meas["rows"]:
            lg, lc = r["latency"]["cuda"], r["latency"]["cpu"]
            lines.append(f"| {NAMES.get(r['variant'], r['variant'])} | {r['params']:,} | {r['train_sec_per_step'] * 1000:.0f} | "
                         f"{r['train_tokens_per_sec']:,.0f} | {r['train_peak_bytes'] / 2**20:.1f} MiB | "
                         f"{lg['mean_ms']:.2f} / {lg['p90_ms']:.2f} ms | {lc['mean_ms']:.2f} / {lc['p90_ms']:.2f} ms | {r['state_bytes_batch1']:,} B |")
    path = OUT / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in summary.items()}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return path
