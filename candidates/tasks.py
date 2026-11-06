"""후보 비교용 과제. 모든 모델이 같은 32개 기호를 쓰고, 같은 관측만 받는다 (모델별 보조 정답 없음).

에피소드 형식
  tokens[t]  : t번째 기호
  score[t]   : True 면 위치 t 에서 낸 예측(= tokens[t+1] 에 대한 예측)을 채점한다
  oracle[t]  : 생성 규칙을 아는 예측 (해당 위치의 최선 답). 없으면 -1

질의 과제(키-값, 관계 사슬)는 답 위치만 채점한다. 흐름 과제는 모든 위치를 채점한다.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------- 기호표 (32개)
VOCAB = 32
DISH = 0          # 0..7   요리 / 값
KEY = 8           # 8..15  키
ENT = 16          # 16..23 개체
STORE, QUERY, ARROW, PAD, SEP = 24, 25, 26, 27, 28
N_SYM = 8
DATA_VERSION = "cand-data-v2"   # v2: 긴 지연 키-값을 회차 구조로 바꿔 지름길(같은 키 연달아 질의)을 없앰
SPLIT_BASE = {"train": 10_000_000, "val": 20_000_000, "test": 30_000_000}   # 학습·검증·시험 난수 집합 분리


@dataclass
class Episode:
    tokens: np.ndarray
    score: np.ndarray
    oracle: np.ndarray
    meta: dict


def _empty(L):
    return np.full(L, PAD, dtype=np.int64), np.zeros(L, dtype=bool), np.full(L, -1, dtype=np.int64)


# ---------------------------------------------------------------- 요리 과제 (요리 8가지의 흐름)
# 생성기는 (요리 열, 시점별 정답 분포, 숨은 교대 횟수) 를 돌려준다. 정답 분포 = 규칙을 아는 예측 = 이론적 최대.
def _regimes(n: int, peak: float, rng: np.random.Generator) -> np.ndarray:
    """요리사 n명의 '직전 요리 → 다음 요리' 표 = 순열 + 잡음. 서로 충분히 다른 순열만 고른다."""
    perms: list[np.ndarray] = []
    max_overlap = max(1, N_SYM // 4)
    while len(perms) < n:
        cand = rng.permutation(N_SYM)
        if all(np.sum(cand == p) <= max_overlap for p in perms):
            perms.append(cand)
    mats = np.full((n, N_SYM, N_SYM), (1.0 - peak) / (N_SYM - 1))
    for k, perm in enumerate(perms):
        mats[k, np.arange(N_SYM), perm] = peak
    return mats


def _schedule(length: int, n: int, dwell: int, rng: np.random.Generator) -> np.ndarray:
    """처음엔 모든 요리사가 한 번씩 근무하고, 이후엔 이미 근무했던 요리사가 무작위로 돌아온다."""
    seq = np.empty(length, dtype=int)
    t, current, visit = 0, -1, 0
    lo, hi = max(5, int(dwell * 0.6)), int(dwell * 1.4)
    while t < length:
        current = visit if visit < n else int(rng.choice([k for k in range(n) if k != current]))
        visit += 1
        span = int(rng.integers(lo, hi + 1))
        seq[t:t + span] = current
        t += span
    return seq


def cook_switching(dwell: int, n_cooks: int = 4, peak: float = 0.8):
    """요리사 n명이 평균 dwell 접시마다 몰래 교대한다."""
    def make(length: int, rng: np.random.Generator):
        mats = _regimes(n_cooks, peak, rng)
        phase = _schedule(length, n_cooks, dwell, rng)
        symbols = np.empty(length, dtype=int)
        oracle = np.empty((length, N_SYM))
        oracle[0] = 1.0 / N_SYM
        symbols[0] = rng.integers(N_SYM)
        for t in range(1, length):
            oracle[t] = mats[phase[t], symbols[t - 1]]
            symbols[t] = rng.choice(N_SYM, p=oracle[t])
        return symbols, oracle, int((phase[1:] != phase[:-1]).sum())
    return make


def cook_phrases(n_phrases: int = 6, eps: float = 0.1):
    """4~8접시짜리 코스 6가지가 무작위 순서로 이어진다. eps 만큼 다른 요리가 섞인다."""
    def make(length: int, rng: np.random.Generator):
        lib = [rng.integers(N_SYM, size=int(rng.integers(4, 9))) for _ in range(n_phrases)]
        firsts = np.zeros(N_SYM)
        for ph in lib:
            firsts[ph[0]] += 1 / n_phrases
        symbols = np.empty(length, dtype=int)
        oracle = np.empty((length, N_SYM))
        k, j = int(rng.integers(n_phrases)), 0
        for t in range(length):
            if j == 0:   # 다음 코스는 아무도 모른다 → 코스 첫 요리들의 섞임
                oracle[t] = (1 - eps) * firsts + eps / N_SYM
            else:
                oracle[t] = eps / N_SYM
                oracle[t, lib[k][j]] += 1 - eps
            true = lib[k][j]
            symbols[t] = true if rng.random() > eps else rng.integers(N_SYM)
            j += 1
            if j == len(lib[k]):
                k, j = int(rng.integers(n_phrases)), 0
        return symbols, oracle, 0
    return make


def cook_cycle(period: int = 24, eps: float = 0.1):
    """정해진 period 접시 순서가 반복된다. eps 만큼 다른 요리가 섞인다."""
    def make(length: int, rng: np.random.Generator):
        seq = rng.integers(N_SYM, size=period)
        symbols = np.empty(length, dtype=int)
        oracle = np.full((length, N_SYM), eps / N_SYM)
        start = int(rng.integers(period))
        for t in range(length):
            true = seq[(start + t) % period]
            oracle[t, true] += 1 - eps
            symbols[t] = true if rng.random() > eps else rng.integers(N_SYM)
        return symbols, oracle, 0
    return make


COOK = {"switch_fast": cook_switching(45), "phrase": cook_phrases(), "cycle": cook_cycle()}


def cook(key: str) -> Callable:
    def make(L: int, rng: np.random.Generator) -> Episode:
        symbols, dist, changes = COOK[key](L, rng)
        tokens = symbols.astype(np.int64) + DISH
        score = np.ones(L, dtype=bool)
        score[-1] = False
        oracle = np.full(L, -1, dtype=np.int64)
        oracle[:-1] = dist[1:].argmax(axis=1) + DISH
        return Episode(tokens, score, oracle, {"changes": changes})
    return make


# ---------------------------------------------------------------- 키-값 저장·덮어쓰기·회수
def key_value(oldest_p: float = 0.0, store_p: float = 0.55) -> Callable:
    """STORE k v 로 저장(같은 키면 덮어쓰기), QUERY k 다음에 현재 값이 나온다. 답 위치(QUERY k 의 k)만 채점.

    oldest_p: 질의할 키를 '가장 오래전에 언급된(저장 또는 질의) 키'로 고를 확률 (지연을 늘리는 변형).
    언급 시점을 기준으로 해야 같은 키를 연달아 물어 바로 앞 답을 베끼는 지름길이 생기지 않는다.
    지연 = 그 키의 값이 마지막으로 기호열에 나타난 뒤 지난 기호 수.
    """
    def make(L: int, rng: np.random.Generator) -> Episode:
        tokens, score, oracle = _empty(L)
        store, when, i, delays = {}, {}, 0, []
        while i + 3 <= L:
            if not store or rng.random() < store_p:
                k, v = int(rng.integers(N_SYM)), int(rng.integers(N_SYM))
                tokens[i:i + 3] = (STORE, KEY + k, DISH + v)
                store[k], when[k] = v, i
            else:
                keys = sorted(store)
                k = min(keys, key=lambda kk: when[kk]) if rng.random() < oldest_p else int(rng.choice(keys))
                tokens[i:i + 3] = (QUERY, KEY + k, DISH + store[k])
                score[i + 1], oracle[i + 1] = True, DISH + store[k]
                delays.append(i - when[k])
                when[k] = i                                  # 질의도 값을 다시 보여 주므로 마지막 언급 시점을 갱신
            i += 3
        return Episode(tokens, score, oracle, {"queries": int(score.sum()), "mean_delay": float(np.mean(delays)) if delays else 0.0,
                                               "max_delay": int(max(delays)) if delays else 0})
    return make


def key_value_long_delay(delay: tuple[int, int] = (120, 300), hidden: int = 2) -> Callable:
    """학습보다 긴 지연. 회차 = 숨길 키를 저장 → 나머지 키로만 하는 저장·질의(방해 구간, delay 기호) → 숨긴 키 질의.

    채점은 숨긴 키의 질의에서만 한다. 키가 8개뿐이라 무작위 질의로는 긴 지연이 생기지 않아 회차 구조로 만든다.
    """
    def make(L: int, rng: np.random.Generator) -> Episode:
        tokens, score, oracle = _empty(L)
        i, delays = 0, []
        while True:
            keys = rng.permutation(N_SYM)
            hid, others = keys[:hidden], keys[hidden:]
            gap = int(rng.integers(delay[0], delay[1] + 1)) // 3 * 3
            if i + 3 * hidden + gap + 3 * hidden > L:
                break
            store, when = {}, {}
            for k in hid:
                v = int(rng.integers(N_SYM))
                tokens[i:i + 3], store[int(k)], when[int(k)] = (STORE, KEY + k, DISH + v), v, i
                i += 3
            for _ in range(gap // 3):                    # 방해 구간: 숨긴 키는 나오지 않는다 (질의는 채점하지 않음)
                k = int(rng.choice(others))
                if k not in store or rng.random() < 0.55:
                    v = int(rng.integers(N_SYM))
                    tokens[i:i + 3], store[k] = (STORE, KEY + k, DISH + v), v
                else:
                    tokens[i:i + 3] = (QUERY, KEY + k, DISH + store[k])
                i += 3
            for k in rng.permutation(hid):
                k = int(k)
                tokens[i:i + 3] = (QUERY, KEY + k, DISH + store[k])
                score[i + 1], oracle[i + 1] = True, DISH + store[k]
                delays.append(i - when[k])
                i += 3
        return Episode(tokens, score, oracle, {"queries": int(score.sum()), "mean_delay": float(np.mean(delays)) if delays else 0.0,
                                               "max_delay": int(max(delays)) if delays else 0})
    return make


def key_value_ladder(gaps: tuple[int, ...], hidden: int = 2, score_all: bool = False) -> Callable:
    """지연 사다리 키-값. 긴 지연 키-값과 같은 회차 구조인데, 방해 구간의 길이를 gaps 에서 고른다.

    숨긴 키의 질의만 채점한다: 방해 구간의 질의까지 채점하면 짧은 지연이 채점의 90% 넘게 차지해 긴 지연의 학습 신호가 묻힌다.
    지연별 성적은 delay_of() 로 위치마다 다시 계산해 구간별로 낸다.
    에피소드 끝에 남은 자리에 안 맞는 지연은 고르지 않는다 (끝부분은 짧은 지연이 조금 더 많다).
    score_all=True 는 학습용: 방해 구간의 질의도 채점해 '키로 값을 찾는' 신호를 늘린다 (기호열은 같은 방식으로 만든다).
    """
    def make(L: int, rng: np.random.Generator) -> Episode:
        tokens, score, oracle = _empty(L)
        i = 0
        while True:
            fit = [g for g in gaps if i + 6 * hidden + g <= L]
            if not fit:
                break
            gap = int(rng.choice(fit)) // 3 * 3
            keys = rng.permutation(N_SYM)
            hid, others = keys[:hidden], keys[hidden:]
            store = {}
            for k in hid:
                v = int(rng.integers(N_SYM))
                tokens[i:i + 3], store[int(k)] = (STORE, KEY + k, DISH + v), v
                i += 3
            for _ in range(gap // 3):                    # 방해 구간: 숨긴 키는 나오지 않는다
                k = int(rng.choice(others))
                if k not in store or rng.random() < 0.55:
                    v = int(rng.integers(N_SYM))
                    tokens[i:i + 3], store[k] = (STORE, KEY + k, DISH + v), v
                else:
                    tokens[i:i + 3] = (QUERY, KEY + k, DISH + store[k])
                    if score_all:
                        score[i + 1], oracle[i + 1] = True, DISH + store[k]
                i += 3
            for k in rng.permutation(hid):
                k = int(k)
                tokens[i:i + 3] = (QUERY, KEY + k, DISH + store[k])
                score[i + 1], oracle[i + 1] = True, DISH + store[k]
                i += 3
        return Episode(tokens, score, oracle, {"queries": int(score.sum())})
    return make


def delay_of(tokens: np.ndarray, score: np.ndarray) -> np.ndarray:
    """채점 위치마다 지연 = 그 키의 값이 마지막으로 기호열에 나온(저장 또는 앞선 질의의 답) 뒤 지난 기호 수. 그 밖은 -1."""
    out = np.full(len(tokens), -1, dtype=np.int64)
    seen = {}
    for i in range(0, len(tokens) - 2, 3):
        op, k = tokens[i], tokens[i + 1]
        if op == QUERY and score[i + 1]:
            out[i + 1] = i + 2 - seen[k]
        if op in (STORE, QUERY):
            seen[k] = i + 2
    return out


def check_key_value(tokens: np.ndarray, score: np.ndarray) -> bool:
    """생성기와 독립적으로 모든 질의의 답을 다시 계산하고, 채점 위치가 질의의 키 위치인지 확인한다."""
    store, query_pos = {}, np.zeros(len(tokens), dtype=bool)
    for i in range(0, len(tokens) - 2, 3):
        op, k, v = tokens[i:i + 3]
        if op == STORE:
            store[k] = v
        elif op == QUERY:
            if store.get(k) != v:
                return False
            query_pos[i + 1] = True
    return bool(score.any() and not (score & ~query_pos).any())


# ---------------------------------------------------------------- 빠른 변화 + 느린 규칙
def mixed_timescales(noise: float = 0.1) -> Callable:
    """기호 = (빠른 주기 패턴 + 느리게 바뀌는 오프셋) mod 8, 10%는 잡음.

    빠른 주기는 3·5·7 중 하나, 오프셋은 40~120 기호마다 바뀐다. 후보의 갱신 주기(1·4·16·64)와 일부러 맞추지 않았다.
    """
    def make(L: int, rng: np.random.Generator) -> Episode:
        P = int(rng.choice([3, 5, 7]))
        pattern, phase = rng.integers(N_SYM, size=P), int(rng.integers(P))
        offset, nxt = int(rng.integers(N_SYM)), int(rng.integers(40, 121))
        clean = np.empty(L, dtype=np.int64)
        for t in range(L):
            if t == nxt:
                offset, nxt = int(rng.integers(N_SYM)), t + int(rng.integers(40, 121))
            clean[t] = (pattern[(phase + t) % P] + offset) % N_SYM
        noisy = np.where(rng.random(L) < noise, rng.integers(N_SYM, size=L), clean)
        tokens = noisy + DISH
        score = np.ones(L, dtype=bool)
        score[-1] = False
        oracle = np.full(L, -1, dtype=np.int64)
        oracle[:-1] = clean[1:] + DISH
        return Episode(tokens, score, oracle, {"period": P})
    return make


# ---------------------------------------------------------------- 관계 사슬 추적
def relation_chain(edges: tuple[int, int] = (1, 3), chains: int = 2, query_head: bool = False) -> Callable:
    """블록 = SEP, 섞인 순서의 사실 'a → b' 들, 질의 'QUERY u 답'. 답은 u 에서 화살표를 끝까지 따라간 개체.

    정답 그래프는 입력으로 따로 주지 않는다. 사실 기호만 본다. query_head=True 면 사슬의 첫 개체를 묻는다(가장 긴 경로).
    """
    def make_block(rng):
        ents = rng.permutation(N_SYM) + ENT
        nxt, facts, heads, used = {}, [], [], 0
        for _ in range(chains):
            n = int(rng.integers(edges[0], edges[1] + 1))
            nodes = ents[used:used + n + 1]
            used += n + 1
            heads.append(nodes)
            for a, b in zip(nodes, nodes[1:]):
                nxt[int(a)] = int(b)
                facts.append((int(a), int(b)))
        rng.shuffle(facts)
        toks, sc, orc = [SEP], [False], [-1]
        for a, b in facts:
            toks += [a, ARROW, b]
            sc += [False] * 3
            orc += [-1] * 3
        starts = [int(n[0]) for n in heads] if query_head else [int(u) for n in heads for u in n[:-1]]
        for u in rng.choice(starts, size=min(len(starts), 3), replace=False):
            end = int(u)
            while end in nxt:
                end = nxt[end]
            toks += [QUERY, int(u), end]
            sc += [False, True, False]
            orc += [-1, end, -1]
        return toks, sc, orc

    def make(L: int, rng: np.random.Generator) -> Episode:
        tokens, score, oracle = _empty(L)
        i = 0
        while True:
            t, s, o = make_block(rng)
            if i + len(t) > L:
                break
            tokens[i:i + len(t)], score[i:i + len(t)], oracle[i:i + len(t)] = t, s, o
            i += len(t)
        return Episode(tokens, score, oracle, {"queries": int(score.sum())})
    return make


def check_relation_chain(tokens: np.ndarray, score: np.ndarray) -> bool:
    nxt, expect, i = {}, np.zeros(len(tokens), dtype=bool), 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == SEP:
            nxt, i = {}, i + 1
        elif tok == QUERY:
            end = tokens[i + 1]
            while end in nxt:
                end = nxt[end]
            if tokens[i + 2] != end:
                return False
            expect[i + 1], i = True, i + 3
        elif tok == PAD:
            i += 1
        else:
            nxt[tokens[i]] = tokens[i + 2]
            i += 3
    return bool((expect == score).all())


# ---------------------------------------------------------------- 새 과제 묶음 (TASK_IDEAS.md 1 · 3 · 4 · 5 · 6)
# 각 생성기는 meta["knob"] 에 위치별 난이도 값(채점 위치만, 나머지 -1)을 남긴다. 결과는 손잡이별 곡선으로 낸다(knobs.py).
ITEM, OPEN_A, OPEN_B, MARK_COPY, MARK_REV = 31, 29, 30, 29, 30
CLOSE_A, CLOSE_B = KEY, KEY + 1


def cyclic_chain(ks: tuple[int, int] = (1, 3), queries: int = 6) -> Callable:
    """k걸음 순환 사슬. 회차마다 개체 8개의 8-순환(끝 개체가 없다)을 사실 8개 'a → b' 로 섞어 보이고,
    'QUERY u k 답' 을 묻는다. 답 = u 에서 k번 따라간 개체. k 는 요리 기호 0–7 을 숫자로 쓴다. k 자리만 채점."""
    def make(L: int, rng: np.random.Generator) -> Episode:
        tokens, score, oracle = _empty(L)
        knob = np.full(L, -1, dtype=np.int64)
        i, per = 0, 1 + 3 * N_SYM + 4 * queries
        while i + per <= L:
            order = rng.permutation(N_SYM)
            succ = np.empty(N_SYM, dtype=np.int64)
            succ[order] = np.roll(order, -1)
            tokens[i] = SEP
            i += 1
            for a in rng.permutation(N_SYM):
                tokens[i:i + 3] = (ENT + a, ARROW, ENT + succ[a])
                i += 3
            for _ in range(queries):
                u, k = int(rng.integers(N_SYM)), int(rng.integers(ks[0], ks[1] + 1))
                v = u
                for _ in range(k):
                    v = int(succ[v])
                tokens[i:i + 4] = (QUERY, ENT + u, DISH + k, ENT + v)
                score[i + 2], oracle[i + 2], knob[i + 2] = True, ENT + v, k
                i += 4
        return Episode(tokens, score, oracle, {"knob": knob, "knob_name": "k"})
    return make


def check_cyclic_chain(tokens, score) -> bool:
    """생성기와 따로: 회차(SEP)마다 사실을 모아 k걸음 따라간 답이 다음 기호와 같은지."""
    succ = {}
    for i in range(len(tokens) - 3):
        if tokens[i] == SEP:
            succ = {}
        elif tokens[i + 1] == ARROW and tokens[i] != QUERY:
            succ[int(tokens[i])] = int(tokens[i + 2])
        if score[i]:
            v = int(tokens[i - 1])
            for _ in range(int(tokens[i]) - DISH):
                v = succ[v]
            if v != tokens[i + 1] or v == tokens[i - 1]:
                return False
    return True


def count_mod(ms: tuple[int, ...] = (2, 3, 5), p_item: float = 0.12, gap: tuple[int, int] = (8, 20)) -> Callable:
    """셈 나머지. 요리 기호가 흐르는 사이에 셀 기호(31)가 가끔 나온다. 가끔 'QUERY m 답' 을 묻고,
    답 = 에피소드 처음부터 나온 셀 기호 수 mod m. m 자리만 채점. 창(64) 밖까지 세야 하는 과제다."""
    def make(L: int, rng: np.random.Generator) -> Episode:
        tokens, score, oracle = _empty(L)
        knob = np.full(L, -1, dtype=np.int64)
        count, i, nxt = 0, 0, int(rng.integers(*gap))
        while i < L:
            if i >= nxt and i + 3 <= L:
                m = int(rng.choice(ms))
                tokens[i:i + 3] = (QUERY, DISH + m, DISH + count % m)
                score[i + 1], oracle[i + 1], knob[i + 1] = True, DISH + count % m, m
                i, nxt = i + 3, i + 3 + int(rng.integers(*gap))
                continue
            if rng.random() < p_item:
                tokens[i] = ITEM
                count += 1
            else:
                tokens[i] = DISH + int(rng.integers(N_SYM))
            i += 1
        return Episode(tokens, score, oracle, {"knob": knob, "knob_name": "m"})
    return make


def check_count_mod(tokens, score) -> bool:
    count = 0
    for i in range(len(tokens) - 1):
        if score[i] and tokens[i + 1] != DISH + count % (int(tokens[i]) - DISH):
            return False
        if tokens[i] == ITEM and not (i >= 2 and tokens[i - 2] == QUERY):
            count += 1
    return True


def copy_reverse(ns: tuple[int, int] = (4, 24)) -> Callable:
    """베끼기 · 뒤집기. 'SEP x1..xn 표시 y1..yn'. 표시 29 = 그대로, 30 = 거꾸로. 기호는 요리 · 키 16가지.
    표시 자리와 y1..y(n-1) 자리(= 다음 y 를 맞힘)를 채점한다. 손잡이 = n."""
    def make(L: int, rng: np.random.Generator) -> Episode:
        tokens, score, oracle = _empty(L)
        knob = np.full(L, -1, dtype=np.int64)
        i = 0
        while True:
            n = int(rng.integers(ns[0], ns[1] + 1))
            if i + 2 * n + 2 > L:
                break
            x = rng.integers(2 * N_SYM, size=n)
            rev = bool(rng.random() < 0.5)
            y = x[::-1] if rev else x
            tokens[i] = SEP
            tokens[i + 1:i + 1 + n] = x
            j = i + 1 + n
            tokens[j] = MARK_REV if rev else MARK_COPY
            tokens[j + 1:j + 1 + n] = y
            score[j:j + n], oracle[j:j + n], knob[j:j + n] = True, y, n
            i = j + 1 + n
        return Episode(tokens, score, oracle, {"knob": knob, "knob_name": "n"})
    return make


def check_copy_reverse(tokens, score) -> bool:
    i = 0
    while i < len(tokens) and tokens[i] == SEP:
        j = i + 1
        while tokens[j] not in (MARK_COPY, MARK_REV):
            j += 1
        x = tokens[i + 1:j]
        y = x[::-1] if tokens[j] == MARK_REV else x
        n = len(x)
        if not ((tokens[j + 1:j + 1 + n] == y).all() and score[j:j + n].all()):
            return False
        i = j + 1 + n
    return True


def bracket_depth(max_depth: int = 4, filler: tuple[int, int] = (0, 3)) -> Callable:
    """괄호 깊이. 두 종류 괄호를 열고(29 · 30), 닫을 때는 'QUERY 닫는기호'(키 0 = A, 키 1 = B) 로 낸다.
    QUERY 자리에서 닫힐 괄호의 종류 = 가장 안쪽에 열린 괄호를 맞힌다. 사이에 요리 기호 0–3개. 손잡이 = 닫기 전 깊이.
    바로 앞 괄호 사건도 닫기였던 자리만 채점한다: 열자마자 닫는 자리는 '가장 최근에 연 괄호' 규칙으로 풀리기 때문이다."""
    def make(L: int, rng: np.random.Generator) -> Episode:
        tokens, score, oracle = _empty(L)
        knob = np.full(L, -1, dtype=np.int64)
        stack, i, prev_close = [], 0, False
        while True:
            f = int(rng.integers(filler[0], filler[1] + 1))
            if i + f + 2 > L:
                break
            tokens[i:i + f] = DISH + rng.integers(N_SYM, size=f)
            i += f
            if stack and (len(stack) == max_depth or rng.random() < 0.5):
                top = stack.pop()
                closer = CLOSE_A if top == OPEN_A else CLOSE_B
                tokens[i:i + 2] = (QUERY, closer)
                if prev_close:
                    score[i], oracle[i], knob[i] = True, closer, len(stack) + 1
                i, prev_close = i + 2, True
            else:
                o = OPEN_A if rng.random() < 0.5 else OPEN_B
                tokens[i] = o
                stack.append(o)
                i, prev_close = i + 1, False
        return Episode(tokens, score, oracle, {"knob": knob, "knob_name": "깊이"})
    return make


def check_bracket_depth(tokens, score) -> bool:
    stack = []
    for i in range(len(tokens) - 1):
        if tokens[i] in (OPEN_A, OPEN_B):
            stack.append(int(tokens[i]))
        elif tokens[i] == QUERY:
            top = stack.pop()
            if tokens[i + 1] != (CLOSE_A if top == OPEN_A else CLOSE_B):
                return False
        elif score[i]:
            return False
    return True


def slow_period(periods: tuple[int, ...] = (32, 64, 128), fast: int = 3, segs: int = 4, noise: float = 0.1) -> Callable:
    """느린 주기. 기호 = (빠른 무늬[t mod 3] + 오프셋(t)) mod 8, 10% 잡음. 오프셋은 P/4 마다 바뀌고,
    네 값이 주기 P 로 되풀이된다(에피소드마다 P 와 값이 다름). 첫 주기가 지나면 오프셋 바뀜도 예측할 수 있다.
    모든 위치를 채점한다. 손잡이 = P, 단 '첫 주기 뒤의 바뀌는 자리'에만 (그 자리는 주기를 기억해야 맞힌다)."""
    def make(L: int, rng: np.random.Generator) -> Episode:
        P = int(rng.choice(periods))
        pattern, offs = rng.integers(N_SYM, size=fast), rng.integers(N_SYM, size=segs)
        while (np.diff(offs) == 0).any() or offs[0] == offs[-1]:      # 바뀌는 자리에서 실제로 값이 바뀌게
            offs = rng.integers(N_SYM, size=segs)
        t = np.arange(L)
        seg = (t % P) // (P // segs)
        clean = (pattern[t % fast] + offs[seg]) % N_SYM
        noisy = np.where(rng.random(L) < noise, rng.integers(N_SYM, size=L), clean)
        score = np.ones(L, dtype=bool)
        score[-1] = False
        oracle = np.full(L, -1, dtype=np.int64)
        oracle[:-1] = clean[1:] + DISH
        knob = np.full(L, -1, dtype=np.int64)
        change = np.zeros(L, dtype=bool)
        change[1:] = seg[1:] != seg[:-1]
        nxt = np.zeros(L, dtype=bool)
        nxt[:-1] = change[1:] & (t[1:] >= P)
        knob[nxt] = P
        return Episode(noisy + DISH, score, oracle, {"knob": knob, "knob_name": "P", "period": P})
    return make


# ---------------------------------------------------------------- 과제 목록
@dataclass
class TaskSpec:
    key: str
    name: str
    kind: str            # stream | query
    length: int
    make: Callable
    skill: str
    train: bool = True   # False 면 시험 전용 (학습보다 긴 일반화)


SPECS: dict[str, TaskSpec] = {s.key: s for s in [
    TaskSpec("switch_fast", "요리사 교대(잦음)", "stream", 256, cook("switch_fast"), "변화 감지"),
    TaskSpec("phrase", "코스 메뉴", "stream", 256, cook("phrase"), "베끼기"),
    TaskSpec("cycle", "24접시 순환", "stream", 256, cook("cycle"), "긴 순서"),
    TaskSpec("kv", "키-값 저장·회수", "query", 256, key_value(), "정확한 기억·덮어쓰기"),
    TaskSpec("mixed", "빠름+느림 혼합", "stream", 256, mixed_timescales(), "두 시간척도"),
    TaskSpec("chain", "관계 사슬", "query", 256, relation_chain(), "합성·반복 계산"),
    TaskSpec("kv_long", "키-값 (긴 지연)", "query", 1024, key_value_long_delay(), "학습보다 긴 지연 (120~300기호)", train=False),
    TaskSpec("chain_long", "관계 사슬 (긴 사슬)", "query", 1024, relation_chain(edges=(4, 7), chains=1, query_head=True), "학습보다 긴 사슬", train=False),
    TaskSpec("mixed_long", "혼합 (긴 시퀀스)", "stream", 1024, mixed_timescales(), "학습보다 긴 시퀀스", train=False),
]}
CORE = tuple(SPECS)                                   # 예비 비교(계획 A–E)의 과제. 데이터 지문의 기본 범위
TRAIN_TASKS = [k for k in CORE if SPECS[k].train]   # 공동 학습 묶음

# 추가 과제: 그 과제를 쓰는 실행의 데이터 지문에만 들어간다 (기존 실행의 판별 키를 바꾸지 않기 위해)
EXTRA_VERSION = "cand-extra-v1"
SPECS.update({s.key: s for s in [
    TaskSpec("kvd", "키-값 지연 사다리", "query", 256, key_value_ladder((6, 12, 24, 48, 96, 144, 192)),
             "64기호를 넘는 지연도 학습에 들어 있음"),
    TaskSpec("kvd_long", "키-값 지연 사다리 (긴 지연)", "query", 1024, key_value_ladder((240, 360, 480, 720, 900)),
             "학습보다 긴 지연 (240~900기호)", train=False),
    TaskSpec("kvd_all", "키-값 지연 사다리 (모든 질의 채점)", "query", 256,
             key_value_ladder((6, 12, 24, 48, 96, 144, 192), score_all=True), "학습용: 방해 구간 질의도 채점"),
]})
# 새 과제 묶음 (계획 H). 학습 5개 + 같은 과제의 더 어려운 시험 5개
SPECS.update({s.key: s for s in [
    TaskSpec("cyc", "순환 사슬", "query", 256, cyclic_chain((1, 3)), "반복 계산 (k = 1–3걸음)"),
    TaskSpec("count", "셈 나머지", "query", 256, count_mod(), "창 밖까지 세기 (mod 2·3·5)"),
    TaskSpec("copy", "베끼기·뒤집기", "query", 256, copy_reverse((4, 24)), "기억 용량 (n = 4–24)"),
    TaskSpec("paren", "괄호 깊이", "query", 256, bracket_depth(4), "쌓아 두기 (깊이 ≤ 4)"),
    TaskSpec("slowp", "느린 주기", "stream", 256, slow_period((32, 64, 128)), "느린 규칙 기억 (P ≤ 128)"),
    TaskSpec("cyc_far", "순환 사슬 (먼 걸음)", "query", 256, cyclic_chain((4, 7)), "학습보다 많은 걸음 (k = 4–7)", train=False),
    TaskSpec("count_long", "셈 나머지 (긴 기록)", "query", 1024, count_mod(), "학습보다 긴 기록 (1024)", train=False),
    TaskSpec("copy_long", "베끼기·뒤집기 (긴 원문)", "query", 256, copy_reverse((28, 48)), "학습보다 긴 원문 (n = 28–48)", train=False),
    TaskSpec("paren_deep", "괄호 깊이 (깊게)", "query", 256, bracket_depth(8), "학습보다 깊게 (깊이 ≤ 8)", train=False),
    TaskSpec("slowp_long", "느린 주기 (긴 주기)", "stream", 1024, slow_period((32, 64, 128, 256)), "학습보다 긴 주기 · 기록", train=False),
]})
SUITE = ("cyc", "count", "copy", "paren", "slowp")
SUITE_VERSION = "suite-v1"   # 새 과제 생성기를 바꾸면 올린다 (캐시 파일 이름 · 판별 키에 들어간다)
SUITE_TESTS = SUITE + ("cyc_far", "count_long", "copy_long", "paren_deep", "slowp_long")
TASK_INDEX = {k: i for i, k in enumerate(SPECS)}     # 추가 과제는 뒤에 붙으므로 기존 번호(=생성 난수)는 그대로


def episode(task: str, split: str, index: int) -> Episode:
    spec = SPECS[task]
    seed = SPLIT_BASE[split] + TASK_INDEX[task] * 100_000 + index
    return spec.make(spec.length, np.random.default_rng(seed))


def dataset(task: str, split: str, n: int, cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    """(n, L) 배열 묶음. cache_dir 가 있으면 한 번 만든 것을 저장해 재사용한다 (판별 키에 버전·개수 포함)."""
    path = None
    if cache_dir is not None:
        ver = f"{DATA_VERSION}_{SUITE_VERSION}" if task in SUITE_TESTS else DATA_VERSION
        path = Path(cache_dir) / f"{task}_{split}_{n}_{ver}.npz"
        if path.exists():
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
    eps = [episode(task, split, i) for i in range(n)]
    out = {"tokens": np.stack([e.tokens for e in eps]), "score": np.stack([e.score for e in eps]),
           "oracle": np.stack([e.oracle for e in eps])}
    if path is not None:   # 다 쓴 뒤 이름을 바꾼다: 여러 프로세스가 반쯤 쓴 파일을 읽지 않게
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
    return out


def describe_tasks(keys=CORE) -> list[dict]:
    return [{"key": s.key, "name": s.name, "kind": s.kind, "length": s.length, "skill": s.skill, "train": s.train}
            for s in (SPECS[k] for k in keys)]


def data_fingerprint(tasks=()) -> str:
    """기본 과제 목록의 지문. tasks 에 추가 과제가 있으면 그 과제의 설명과 버전도 넣는다."""
    parts = [DATA_VERSION, describe_tasks()]
    extra = sorted(set(tasks) - set(CORE))
    if extra:
        parts.append([EXTRA_VERSION, describe_tasks(extra)])
    if set(tasks) & set(SUITE_TESTS):
        parts.append(SUITE_VERSION)
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]
