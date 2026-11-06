"""후보 모델: Folio · Strata · Lattice 와 기준 모델 GRU · Transformer.

공통 계약 (ARCHITECTURE_CANDIDATES.md §2)
  initial_state(batch_size, device) -> dict[str, Tensor]   에피소드 시작 상태 (0)
  step(x_t, state) -> (logits_next, new_state)              x_t 를 관측한 뒤 x_(t+1) 을 예측
  forward(tokens, state=None) -> (logits, state)            step 을 순서대로 실행한 것과 같은 결과

상태는 텐서 사전이다. detach_state / state_bytes 로 모든 모델을 같은 방식으로 다룬다.
학습 가능한 슬롯 주소·셀 위치 벡터는 가중치이며 상태가 아니다.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

State = dict[str, torch.Tensor]


def detach_state(state: State) -> State:
    return {k: v.detach() for k, v in state.items()}


def state_bytes(state: State) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values()))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Candidate(nn.Module):
    """공통 틀. 기본 forward 는 step 을 순서대로 실행한다 (정의상 step 누적과 같다)."""

    def initial_state(self, batch_size: int, device) -> State:
        raise NotImplementedError

    def _dtype(self):
        return next(self.parameters()).dtype

    def step(self, x_t: torch.Tensor, state: State) -> tuple[torch.Tensor, State]:
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor, state: State | None = None) -> tuple[torch.Tensor, State]:
        if state is None:
            state = self.initial_state(tokens.shape[0], tokens.device)
        self.begin_aux()
        out = []
        for t in range(tokens.shape[1]):
            logits, state = self.step(tokens[:, t], state)
            out.append(logits)
        return torch.stack(out, dim=1), state

    # 보조 손실이 없는 모델은 0
    def begin_aux(self):
        pass

    def aux_loss(self) -> torch.Tensor | None:
        return None


class Head(nn.Module):
    def __init__(self, d: int, vocab: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)

    def forward(self, h):
        return self.out(self.ln(h))


# ---------------------------------------------------------------- 기준 모델: GRU
@dataclass
class GRUConfig:
    vocab: int = 32
    d_embed: int = 64
    d_hidden: int = 150


class GRUBaseline(Candidate):
    """일반 GRU. 학습 속도를 위해 forward 는 nn.GRU 를 한 번에 돌린다 (step 누적과의 일치는 검증에서 확인)."""

    def __init__(self, cfg: GRUConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_embed)
        self.gru = nn.GRU(cfg.d_embed, cfg.d_hidden, batch_first=True)
        self.head = Head(cfg.d_hidden, cfg.vocab)

    def initial_state(self, batch_size, device):
        return {"h": torch.zeros(batch_size, self.cfg.d_hidden, device=device, dtype=self._dtype())}

    def step(self, x_t, state):
        out, h = self.gru(self.embed(x_t)[:, None], state["h"][None])
        return self.head(out[:, 0]), {"h": h[0]}

    def forward(self, tokens, state=None):
        if state is None:
            state = self.initial_state(tokens.shape[0], tokens.device)
        out, h = self.gru(self.embed(tokens), state["h"][None])
        return self.head(out), {"h": h[0]}


# ---------------------------------------------------------------- 기준 모델: Transformer (GPT형)
@dataclass
class TransformerBaselineConfig:
    vocab: int = 32
    context: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2


class Block(nn.Module):
    """사전 정규화 블록: 인과 자기주의 + MLP(4배)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.heads = heads
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """토큰 임베딩 + 학습되는 위치 임베딩 + 블록 n개 + LayerNorm + 출력층."""

    def __init__(self, cfg: TransformerBaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


class TransformerBaseline(Candidate):
    """GPT형 Transformer. 상태 = 최근 context 개의 토큰(창). KV 캐시는 구현하지 않았다.

    step 은 매번 창 전체를 다시 계산한다. 창 안의 위치 번호는 창 시작부터 0, 1, 2 … 이다.
    창이 다 차면 가장 오래된 토큰을 버린다. 따라서 이 모델의 '지속 기억'은 최근 context 토큰까지다.
    """

    def __init__(self, cfg: TransformerBaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.gpt = TinyGPT(cfg)

    def initial_state(self, batch_size, device):
        return {"buf": torch.zeros(batch_size, self.cfg.context, dtype=torch.long, device=device),
                "len": torch.zeros(batch_size, dtype=torch.long, device=device)}

    @staticmethod
    def _length(state) -> int:
        n = state["len"]
        assert bool((n == n[0]).all()), "배치 안의 모든 에피소드가 같은 시점이어야 한다"
        return int(n[0])

    def _append(self, state, tokens):
        C, n = self.cfg.context, self._length(state)
        seq = torch.cat([state["buf"][:, C - n:] if n else state["buf"][:, :0], tokens], dim=1)[:, -C:]
        buf = torch.cat([torch.zeros(seq.shape[0], C - seq.shape[1], dtype=torch.long, device=seq.device), seq], dim=1)
        return {"buf": buf, "len": torch.full_like(state["len"], seq.shape[1])}

    def step(self, x_t, state):
        state = self._append(state, x_t[:, None])
        n = int(state["len"][0])
        return self.gpt(state["buf"][:, -n:])[:, -1], state

    def forward(self, tokens, state=None):
        if state is None:
            state = self.initial_state(tokens.shape[0], tokens.device)
        n, T = self._length(state), tokens.shape[1]
        if n + T <= self.cfg.context:   # 창을 넘지 않으면 한 번에 계산해도 step 누적과 같다
            seq = torch.cat([state["buf"][:, self.cfg.context - n:], tokens], dim=1) if n else tokens
            return self.gpt(seq)[:, n:], self._append(state, tokens)
        return super().forward(tokens, state)   # 창이 밀리는 구간은 step 을 그대로 반복한다


# ---------------------------------------------------------------- Folio
@dataclass
class FolioConfig:
    vocab: int = 32
    d_embed: int = 64
    d_ctrl: int = 64       # 제어 상태 h
    slots: int = 32        # 기억 슬롯 수
    d_slot: int = 64       # 슬롯당 차원
    d_key: int = 32        # 주소 차원
    reads: int = 2         # 입력당 읽기·계산 반복 R
    memory: bool = True    # False = 기억 읽기·쓰기 제거 대조군
    write_gate: bool = False   # v2: 쓰기 세기 a∈(0,1). False 면 매 입력 반드시 어딘가에 쓴다 (v1)
    content: bool = False      # v2: 주소 = 고정 주소 일치 + β·(질의와 칸 내용의 코사인). 내용으로 칸을 찾는다


class Folio(Candidate):
    """주소가 있는 연속값 기억 + 순환 제어기 (§3). 슬롯 선택에 softmax 주소 어텐션을 쓴다."""

    def __init__(self, cfg: FolioConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_embed)
        self.gru_in = nn.GRUCell(cfg.d_embed, cfg.d_ctrl)
        self.gru_cmp = nn.GRUCell(cfg.d_embed + cfg.d_slot, cfg.d_ctrl)
        self.keys = nn.Parameter(torch.randn(cfg.slots, cfg.d_key))     # 슬롯마다 다른 난수 주소
        self.w_read = nn.Linear(cfg.d_ctrl, cfg.d_key, bias=False)
        self.w_write = nn.Linear(cfg.d_ctrl, cfg.d_key, bias=False)
        self.w_erase = nn.Linear(cfg.d_ctrl, cfg.d_slot)
        self.w_value = nn.Linear(cfg.d_ctrl, cfg.d_slot)
        if cfg.write_gate:
            self.w_gate = nn.Linear(cfg.d_ctrl, 1)
            nn.init.constant_(self.w_gate.bias, -1.0)               # 처음에는 약하게 쓴다
        if cfg.content:                                             # 읽기 · 쓰기마다 (내용 질의, 세기 β)
            self.w_cread = nn.Linear(cfg.d_ctrl, cfg.d_slot + 1)
            self.w_cwrite = nn.Linear(cfg.d_ctrl, cfg.d_slot + 1)
        self.head = Head(cfg.d_ctrl, cfg.vocab)

    def initial_state(self, batch_size, device):
        c = self.cfg
        return {"h": torch.zeros(batch_size, c.d_ctrl, device=device, dtype=self._dtype()),
                "M": torch.zeros(batch_size, c.slots, c.d_slot, device=device, dtype=self._dtype())}

    def _address(self, q, h=None, M=None, proj=None):
        score = q @ self.keys.t() / math.sqrt(self.cfg.d_key)                         # (B, S) 고정 주소
        if self.cfg.content and proj is not None:
            cq = proj(h)
            qc, beta = cq[:, :-1], F.softplus(cq[:, -1:])
            cos = torch.einsum("bd,bsd->bs", qc, M) / (qc.norm(dim=-1, keepdim=True) * M.norm(dim=-1) + 1e-6)
            score = score + beta * cos                                              # 빈 칸(M=0)은 0
        return torch.softmax(score, dim=-1)

    def step(self, x_t, state):
        c = self.cfg
        e = self.embed(x_t)
        h, M = state["h"], state["M"]
        h = self.gru_in(e, h)
        for _ in range(c.reads):                       # 읽기 동안 기억은 M_previous 로 고정
            if c.memory:
                r = torch.einsum("bs,bsd->bd", self._address(self.w_read(h), h, M, getattr(self, "w_cread", None)), M)
            else:
                r = torch.zeros(e.shape[0], c.d_slot, device=e.device, dtype=e.dtype)
            h = self.gru_cmp(torch.cat([e, r], dim=-1), h)
        if c.memory:                                   # 쓰기는 마지막에 한 번, 모든 슬롯을 동시에
            w = self._address(self.w_write(h), h, M, getattr(self, "w_cwrite", None))[..., None]   # (B, S, 1)
            if c.write_gate:
                w = w * torch.sigmoid(self.w_gate(h))[:, :, None]       # 쓰지 않기로 할 수 있다
            g = torch.sigmoid(self.w_erase(h))[:, None, :]                # (B, 1, D)
            v = torch.tanh(self.w_value(h))[:, None, :]
            M = (1 - w * g) * M + w * v
        return self.head(h), {"h": h, "M": M}


# ---------------------------------------------------------------- Strata
@dataclass
class StrataConfig:
    vocab: int = 32
    d: int = 64                                  # 수준별 상태 차원 (= 입력 임베딩 차원)
    periods: tuple = (1, 4, 16, 64)              # 수준별 갱신 주기
    aux_weight: float = 0.05                     # 보조 예측 손실 계수 lambda (0 = 제거 대조군)


class Strata(Candidate):
    """서로 다른 주기로 갱신되는 네 수준의 예측 상태 (§4).

    시계 t 는 에피소드 시작부터 센다. 상태 안에 들어 있으므로 청크 경계에서 초기화되지 않는다.
    """

    def __init__(self, cfg: StrataConfig):
        super().__init__()
        self.cfg = cfg
        L, d = len(cfg.periods), cfg.d
        self.embed = nn.Embedding(cfg.vocab, d)
        self.grus = nn.ModuleList(nn.GRUCell(3 * d, d) for _ in range(L))          # 입력 = (z, err, c)
        self.preds = nn.ModuleList(nn.Linear(2 * d, d) for _ in range(L))          # P_l(s_l, c_l)
        self.head = Head(L * d, cfg.vocab)
        self._aux_sum = self._aux_n = None

    def initial_state(self, batch_size, device):
        L, d = len(self.cfg.periods), self.cfg.d
        dt = self._dtype()
        st = {f"s{l}": torch.zeros(batch_size, d, device=device, dtype=dt) for l in range(L)}
        st |= {f"sum{l}": torch.zeros(batch_size, d, device=device, dtype=dt) for l in range(1, L)}
        st |= {f"cnt{l}": torch.zeros(batch_size, 1, device=device, dtype=dt) for l in range(1, L)}
        st["t"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        return st

    def begin_aux(self):
        L = len(self.cfg.periods)
        self._aux_sum = [0.0] * L
        self._aux_n = [0.0] * L

    def aux_loss(self):
        """수준별로 실제 갱신된 항의 평균 제곱 오차를 구한 뒤, 갱신이 있었던 수준끼리 다시 평균한다.
        CPU 동기화 없이 계산한다(CUDA graph 캡처용): 갱신이 없던 수준은 가중치 0 으로 뺀다."""
        ref = next((x for x in self._aux_sum or () if torch.is_tensor(x)), None)
        if ref is None or self.cfg.aux_weight <= 0:
            return None
        dev, dt = ref.device, ref.dtype
        s = torch.stack([torch.as_tensor(x, device=dev, dtype=dt) for x in self._aux_sum])
        n = torch.stack([torch.as_tensor(x, device=dev, dtype=dt) for x in self._aux_n])
        has = (n > 0).to(dt)
        return (s / n.clamp(min=1) * has).sum() / has.sum().clamp(min=1)

    def forward(self, tokens, state=None):
        """배치의 모든 에피소드가 같은 시계면 어느 수준이 갱신되는지 미리 알 수 있다 → 갱신하지 않는 수준은 계산을 건너뛴다.
        건너뛴 수준의 결과는 '상태 유지'로 마스크 경로와 같다 (검증의 forward = step 누적으로 확인)."""
        if state is None:
            state = self.initial_state(tokens.shape[0], tokens.device)
        t = state["t"]
        if getattr(self, "masked_path", False):   # 묶음 학습(fast.py): 시계를 CPU 로 읽지 않고 마스크 경로만 쓴다
            clock = None
        else:
            clock = int(t[0]) if bool((t == t[0]).all()) else None
        self.begin_aux()
        out = []
        for i in range(tokens.shape[1]):
            logits, state = self._step(tokens[:, i], state, None if clock is None else clock + i)
            out.append(logits)
        return torch.stack(out, dim=1), state

    def step(self, x_t, state):
        return self._step(x_t, state, None)

    def _step(self, x_t, state, clock):
        cfg, L = self.cfg, len(self.cfg.periods)
        e = self.embed(x_t)
        t = state["t"]
        old = [state[f"s{l}"] for l in range(L)]
        ctx = old[1:] + [torch.zeros_like(old[0])]          # 상위 문맥: 이번 토큰 처리 전의 s_(l+1), 최상위는 0
        new, out = list(old), dict(state)
        for l in range(L):
            if l == 0:
                z = e
                upd = torch.ones_like(t, dtype=torch.bool)
            else:
                acc = state[f"sum{l}"] + new[l - 1]           # 이번 토큰에서의 하위 상태 (갱신됐으면 새 값)
                cnt = state[f"cnt{l}"] + 1
                upd = (t + 1) % cfg.periods[l] == 0
                z = acc / cnt
            if cfg.periods[l] == 1:
                upd = torch.ones_like(t, dtype=torch.bool)
            if clock is not None and cfg.periods[l] > 1 and (clock + 1) % cfg.periods[l] != 0:
                out[f"sum{l}"], out[f"cnt{l}"] = acc, cnt       # 이번 토큰에 갱신하지 않는 수준: 누적만 한다
                continue
            pred = self.preds[l](torch.cat([old[l], ctx[l]], dim=-1))
            err = z - pred
            cand = self.grus[l](torch.cat([z, err, ctx[l]], dim=-1), old[l])
            m = upd[:, None]
            new[l] = torch.where(m, cand, old[l])
            if l > 0:
                out[f"sum{l}"] = torch.where(m, torch.zeros_like(acc), acc)
                out[f"cnt{l}"] = torch.where(m, torch.zeros_like(cnt), cnt)
            if self._aux_sum is not None and cfg.aux_weight > 0:
                se = ((pred - z.detach()) ** 2).mean(dim=-1)                 # 행별 평균 제곱 오차
                self._aux_sum[l] = self._aux_sum[l] + (se * upd).sum()
                self._aux_n[l] = self._aux_n[l] + upd.sum()
        out |= {f"s{l}": new[l] for l in range(L)}
        out["t"] = t + 1
        return self.head(torch.cat(new, dim=-1)), out


# ---------------------------------------------------------------- Lattice
@dataclass
class LatticeConfig:
    vocab: int = 32
    d_embed: int = 64
    side: int = 4             # side × side 토러스
    d_cell: int = 32          # 셀당 상태 차원
    d_pos: int = 8            # 셀 위치 벡터 차원
    iters: int = 3            # 입력당 내부 반복 K
    neighbors: bool = True    # False = 이웃 메시지를 0으로 만든 대조군
    inject_all: bool = False  # v2: 입력을 모든 셀에 넣는다 (셀마다 배운 세기). False 면 0번 셀에만 (v1)
    norm: bool = False        # v2: 이웃 메시지를 LayerNorm 해서 GRU 에 넣는다 (학습 안정)


class Lattice(Candidate):
    """고정 격자 위의 공유 계산 (§5). 모든 셀은 같은 반복 시작 상태에서 읽고 동시에 갱신한다."""

    def __init__(self, cfg: LatticeConfig):
        super().__init__()
        self.cfg = cfg
        n, d = cfg.side * cfg.side, cfg.d_cell
        self.embed = nn.Embedding(cfg.vocab, cfg.d_embed)
        self.w_in = nn.Linear(cfg.d_embed, d, bias=False)
        self.w_dir = nn.ModuleDict({k: nn.Linear(d, d, bias=False) for k in ("up", "down", "left", "right")})
        self.pos = nn.Parameter(torch.randn(n, cfg.d_pos) * 0.1)
        self.cell = nn.GRUCell(d + cfg.d_pos, d)
        if cfg.inject_all:
            self.inject = nn.Parameter(torch.cat([torch.ones(1, 1), torch.full((n - 1, 1), 0.1)]))   # 셀별 입력 세기
        if cfg.norm:
            self.msg_ln = nn.LayerNorm(d)
        self.head = Head(d, cfg.vocab)
        r, c = torch.arange(n) // cfg.side, torch.arange(n) % cfg.side
        s = cfg.side
        nbr = [((r - 1) % s) * s + c, ((r + 1) % s) * s + c, r * s + (c - 1) % s, r * s + (c + 1) % s]  # 상·하·좌·우
        self.register_buffer("idx_nbr", torch.stack(nbr, dim=1), persistent=False)                      # (n, 4)

    def initial_state(self, batch_size, device):
        c = self.cfg
        return {"H": torch.zeros(batch_size, c.side * c.side, c.d_cell, device=device, dtype=self._dtype())}

    def step(self, x_t, state):
        c = self.cfg
        H = state["H"]
        B, n, d = H.shape
        if c.inject_all:
            H = H + self.inject[None] * self.w_in(self.embed(x_t))[:, None]                 # 모든 셀, 셀별 세기
        else:
            H = torch.cat([H[:, :1] + self.w_in(self.embed(x_t))[:, None], H[:, 1:]], dim=1)   # 0번 셀에만 입력 주입
        pos = self.pos[None].expand(B, n, c.d_pos)
        # m_i = W_up H_up(i) + W_down H_down(i) + W_left H_left(i) + W_right H_right(i) 를 이어 붙인 가중치 한 번으로 계산
        w_cat = torch.cat([self.w_dir[k].weight for k in ("up", "down", "left", "right")], dim=1)       # (d, 4d)
        for _ in range(c.iters):
            if c.neighbors:                                    # 모두 같은 H 에서 읽는다 → 같은 반복 안의 추가 전파 없음
                m = H[:, self.idx_nbr].reshape(B, n, 4 * d) @ w_cat.t()
            else:
                m = torch.zeros_like(H)
            if c.norm:
                m = self.msg_ln(m)
            H = self.cell(torch.cat([m, pos], dim=-1).reshape(B * n, -1), H.reshape(B * n, d)).reshape(B, n, d)
        return self.head(H.mean(dim=1)), {"H": H}


# ---------------------------------------------------------------- 만들기
# 약 10만 파라미터(±10%)에 맞추려고 시작안에서 바꾼 폭. 이유는 CANDIDATES.md 의 '크기 맞추기' 절에 기록한다.
SIZE_TUNING = {
    "transformer": {},                                  # d=64, 2층, 문맥 64 그대로
    "gru": {"d_hidden": 150},
    "folio": {"d_ctrl": 76},                            # 64 → 76
    "strata": {"d": 40},                                # 64 → 40 (수준 4개 × GRU 입력 3d 라 64면 약 23만)
    "lattice": {"d_cell": 92},                          # 32 → 92 (32면 약 1.8만)
}
CONFIGS = {"transformer": TransformerBaselineConfig, "gru": GRUConfig, "folio": FolioConfig,
           "strata": StrataConfig, "lattice": LatticeConfig}
CLASSES = {"transformer": TransformerBaseline, "gru": GRUBaseline, "folio": Folio, "strata": Strata, "lattice": Lattice}

# 핵심 대조군 (§3–§5). 이름 → (기본 모델, 설정 변경)
VARIANTS = {
    "transformer": ("transformer", {}),
    "gru": ("gru", {}),
    "folio": ("folio", {}),
    "strata": ("strata", {}),
    "lattice": ("lattice", {}),
    "folio_nomem": ("folio", {"memory": False}),
    "folio_r1": ("folio", {"reads": 1}),
    "strata_sameperiod": ("strata", {"periods": (1, 1, 1, 1)}),
    "strata_noaux": ("strata", {"aux_weight": 0.0}),
    "lattice_k1": ("lattice", {"iters": 1}),
    "lattice_nonbr": ("lattice", {"neighbors": False}),
    # 개선판 (v2). 약 10만 파라미터에 맞추려고 Folio 는 제어 상태를 76 → 72 로 줄였다
    "folio2": ("folio", {"write_gate": True, "content": True, "d_ctrl": 72}),
    "lattice2": ("lattice", {"inject_all": True, "norm": True}),
    # 창 256 Transformer: 학습 에피소드(256) 전체를 한 번에 본다. 위치 임베딩이 늘어 파라미터 약 12만
    "transformer256": ("transformer", {"context": 256}),
}


def build(variant: str, vocab: int, sized: bool = True) -> Candidate:
    base, change = VARIANTS[variant]
    cfg = CONFIGS[base](vocab=vocab)
    if sized:
        cfg = replace(cfg, **SIZE_TUNING[base])
    cfg = replace(cfg, **change)
    return CLASSES[base](cfg)


# 나중에 더한 설정 항목과 그 기본값(= 예전 동작). 기본값이면 설명에서 빼서, 예전 실행의 판별 키가 바뀌지 않게 한다
ADDED_FIELDS = {"write_gate": False, "content": False, "inject_all": False, "norm": False}


def describe(model: Candidate) -> dict:
    cfg = {k: v for k, v in asdict(model.cfg).items() if not (k in ADDED_FIELDS and v == ADDED_FIELDS[k])}
    return {"class": type(model).__name__, "params": n_params(model), "config": cfg}
