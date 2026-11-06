"""main.py 가 채우는 실험 계획. 묶음마다 설정 목록을 만든다 (설정이 같으면 실행 이름도 같아 결과를 재사용한다).

  A  공동 학습 · 지속 기억 · 기본 5개 × 난수 3 · 800걸음
  B  공동 학습 · 창 초기화 · 기본 5개 × 난수 3 · 800걸음
  C  공동 학습 · 지속 기억 · 대조군 6개 × 난수 3 · 800걸음
  D  과제별 학습(키-값 · 혼합 · 관계 사슬) · 지속 기억 · 기본 5개 × 난수 3 · 400걸음
  E  공동 학습 · 지속 기억 · 기본 5개 + 개선판 2개 × 난수 3 · 3000걸음 (충분히 학습한 비교. 처음엔 난수 0 만 진단으로 돌렸다)
  F  지연 사다리 키-값만 · 지속 기억 · Folio · GRU × 청크 64/256 × 난수 3 + Transformer(청크 64) · 2000걸음
     + 나머지 후보와 개선판(Strata · Lattice · Folio2 · Lattice2)은 청크 256 만
     청크 256 = 에피소드 전체에 기울기가 닿는다. 64 를 넘는 기억을 '배울 기회'가 있을 때와 없을 때를 가른다.
  G  F 와 같은데 학습만 방해 구간 질의까지 채점(kvd_all). 시험은 F 와 같다(숨긴 키 질의만).
     F 의 Transformer 는 채점이 회차당 2개뿐이라 창 안에서도 조회를 배우지 못했다 → 모든 모델에 같은 학습 신호를 준다.
  H  새 과제 묶음(순환 사슬 · 셈 나머지 · 베끼기·뒤집기 · 괄호 깊이 · 느린 주기) 공동 학습 · 3000걸음 · 난수 3.
     순환형은 청크 256(에피소드 전체에 기울기), Transformer 는 창 64 와 창 256 둘 다. 시험은 각 과제의 더 어려운 판까지.
     Strata 의 '주기를 나눈 층' 가설은 느린 주기 과제에서 '모든 층 매번 갱신' 대조군과 비교해 판단한다.
"""
from __future__ import annotations

from .benchmark import RunConfig
from .tasks import SUITE, SUITE_TESTS, TRAIN_TASKS

MAIN = ["transformer", "gru", "folio", "strata", "lattice"]
ABLATIONS = ["folio_nomem", "folio_r1", "strata_sameperiod", "strata_noaux", "lattice_k1", "lattice_nonbr"]
IMPROVED = ["folio2", "lattice2"]          # 개선판: 예비 비교에서 본 약점을 고친 것 (models.py VARIANTS)
SEEDS = (0, 1, 2)
COMMON = {"batch": 64, "n_test": 128}

GROUPS = {
    "A": "공동 · 지속 기억 · 800걸음",
    "B": "공동 · 창 초기화 · 800걸음",
    "C": "대조군 · 공동 · 지속 기억 · 800걸음",
    "D": "과제별 · 지속 기억 · 400걸음",
    "E": "공동 · 지속 기억 · 3000걸음 · 개선판 포함",
    "F": "지연 사다리 키-값 · 청크 64/256 · 2000걸음",
    "G": "지연 사다리 키-값 · 모든 질의 채점 학습 · 2000걸음",
    "H": "새 과제 묶음 · 공동 · 3000걸음",
}


def configs(group: str) -> list[RunConfig]:
    joint = tuple(TRAIN_TASKS)
    # 난수별로 모델을 번갈아 둔다: 병렬 실행 때 메모리를 많이 쓰는 Lattice 가 한꺼번에 몰리지 않게
    if group == "A":
        return [RunConfig(v, joint, "persistent", s, steps=800, eval_every=200, **COMMON) for s in SEEDS for v in MAIN]
    if group == "B":
        return [RunConfig(v, joint, "window", s, steps=800, eval_every=200, **COMMON) for s in SEEDS for v in MAIN]
    if group == "C":
        return [RunConfig(v, joint, "persistent", s, steps=800, eval_every=200, **COMMON) for s in SEEDS for v in ABLATIONS]
    if group == "D":
        return [RunConfig(v, (t,), "persistent", s, steps=400, eval_every=100, **COMMON)
                for t in ("kv", "mixed", "chain") for s in SEEDS for v in MAIN]
    if group == "E":
        return [RunConfig(v, joint, "persistent", s, steps=3000, eval_every=500, **COMMON) for s in SEEDS for v in MAIN + IMPROVED]
    if group == "F":
        # 오래 걸리는 Folio 를 앞에 둔다: 병렬 실행이 끝까지 고르게 바쁘도록
        f = dict(steps=2000, eval_every=250, batch=64, n_test=256)
        return ([RunConfig(v, ("kvd",), "persistent", s, chunk=c, **f) for v in ("folio", "gru") for c in (256, 64) for s in SEEDS]
                + [RunConfig("transformer", ("kvd",), "persistent", s, chunk=64, **f) for s in SEEDS]
                + [RunConfig(v, ("kvd",), "persistent", s, chunk=256, **f) for v in ("strata", "lattice") + tuple(IMPROVED) for s in SEEDS])
    if group == "G":
        f = dict(steps=2000, eval_every=250, batch=64, n_test=256, eval_tasks=("kvd", "kvd_long"))
        return ([RunConfig(v, ("kvd_all",), "persistent", s, chunk=c, **f) for v in ("folio", "gru") for c in (256, 64) for s in SEEDS]
                + [RunConfig("transformer", ("kvd_all",), "persistent", s, chunk=64, **f) for s in SEEDS])
    if group == "H":
        h = dict(steps=3000, eval_every=500, batch=64, n_test=128, eval_tasks=SUITE_TESTS)
        chunk = {"transformer": 64, "transformer256": 256}   # 트랜스포머는 학습 청크 = 창 (benchmark.run_episodes)
        return [RunConfig(v, SUITE, "persistent", s, chunk=chunk.get(v, 256), **h)
                for s in SEEDS for v in MAIN + IMPROVED + ["transformer256", "strata_sameperiod"]]
    raise KeyError(group)


def all_configs() -> list[RunConfig]:
    return [c for g in GROUPS for c in configs(g)]
