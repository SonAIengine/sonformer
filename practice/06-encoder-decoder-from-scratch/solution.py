"""
Encoder-Decoder Transformer — original 2017 paper ("Attention is All You Need").
Single-file reference (정답지) for hand-coding exercises in this folder.

Key difference from `practice/05-sonlm-from-scratch/solution.py`:
    - Two stacks: Encoder (bidirectional self-attn) + Decoder (causal self-attn + cross-attn)
    - Decoder layers have THREE sub-layers (self-attn, cross-attn, FFN)
    - Sinusoidal positional encoding (paper-original) instead of learned PE
    - Designed for seq2seq (translation, summarization, etc.) — here demoed via
      a sequence-reversal task: src=[a,b,c,d,EOS] → tgt=[BOS,d,c,b,a,EOS]

읽는 순서 (학습용):
    Config → sinusoidal_pe → MultiHeadAttention → FFN
    → EncoderLayer (2 sub-layer) → DecoderLayer (3 sub-layer ★ 핵심)
    → Transformer (encode/decode/generate)
    → mask 유틸 → reverse 데이터셋 → smoke_test 루프

Run:
    python solution.py        # 약 1500 step에 reverse task 학습 (~30s on H100)
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────
# Config — 모든 하이퍼파라미터 한 곳에
# ───────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Vocab + 특수 토큰 ─────────────────────────────────────────────
    # 모델이 다루는 모든 입력은 0 ~ vocab_size-1 범위의 정수 ID.
    # 그 중 앞 3개는 "특수 토큰"으로 예약하고, 나머지 3~31이 실제 어휘.
    #   ID:  0      1      2      3, 4, ..., 31
    #        PAD    BOS    EOS    └─ 실제 토큰 ─┘
    vocab_size: int = 32   # 임베딩 테이블 행 수 = 마지막 softmax 클래스 수
    pad_id: int = 0        # 길이 맞추기용 빈칸. attention/loss에서 무시됨
    bos_id: int = 1        # "출력 시작" 신호. decoder의 generate 첫 입력
    eos_id: int = 2        # "출력 끝" 신호. generate 루프 종료 조건

    # Model
    d_model: int = 128             # 토큰 한 개의 hidden 차원
    n_heads: int = 4               # multi-head attention 헤드 수 (head_dim = d_model/n_heads = 32)
    n_layers: int = 3              # encoder layers = decoder layers (편의상 동일)
    ffn_hidden: int = 512          # FFN 내부 차원, 보통 4× d_model (paper 원본)
    max_seq_len: int = 16          # 이 task에서 시퀀스가 짧으니 16이면 충분
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    warmup_steps: int = 100        # 처음 100 step은 LR을 0→3e-4로 선형 증가 (불안정 회피)
    max_steps: int = 1500
    weight_decay: float = 0.01
    grad_clip: float = 1.0         # gradient norm 1.0으로 clip → 폭발 방지
    label_smoothing: float = 0.1   # 정답을 1.0이 아닌 0.9로 (overconfidence 완화, paper trick)
    log_interval: int = 100
    seed: int = 42


# ───────────────────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding (paper-original)
# ───────────────────────────────────────────────────────────────────────

def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """
    Paper의 PE 정의:
        PE_(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE_(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    직관:
        각 차원 i 마다 서로 다른 주파수의 sin/cos를 깔아서, 모든 (pos, dim)
        조합이 고유한 패턴을 갖게 한다. 학습되지 않는 고정값(buffer)이라
        학습 데이터에서 본 적 없는 길이에도 잘 외삽되는 게 장점.

    구현 트릭:
        나누기를 직접 하면 큰 지수에서 수치 불안정.
        대신 log → exp 로 변환: 10000^(2i/d) = exp(log(10000) · 2i/d).

    Returns: (max_len, d_model) — 학습 안 함, register_buffer 로 모델에 붙임.
    """
    pos = torch.arange(max_len).unsqueeze(1).float()                  # (max_len, 1)  — 위치
    i = torch.arange(0, d_model, 2).float()                            # (d_model/2,)  — 짝수 차원 인덱스
    div = torch.exp(-math.log(10000.0) * i / d_model)                  # (d_model/2,)  — 1/10000^(2i/d)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)                                 # 짝수 차원 = sin
    pe[:, 1::2] = torch.cos(pos * div)                                 # 홀수 차원 = cos
    return pe


# ───────────────────────────────────────────────────────────────────────
# Multi-Head Attention — general (supports self- AND cross-attention)
# ───────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    하나의 attention 클래스로 self / cross 둘 다 처리. 차이는 *입력* 만:

        self-attention:   q_in == k_in == v_in   (같은 시퀀스를 본다)
        cross-attention:  q_in = decoder hidden, k_in = v_in = encoder output
                          (decoder가 encoder를 '쳐다본다')

    decoder-only LM과 달리 여기선 q와 k/v 길이가 다를 수 있으므로
    Tq, Tk를 분리해서 다룬다 (cross-attn에서 Tq != Tk).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        # Q, K, V projection을 각각 따로 (cross-attn에서는 입력 텐서가 다르므로 fused QKV 불가)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Tq, C = q_in.shape          # query 시퀀스 길이
        Tk = k_in.shape[1]             # key/value 시퀀스 길이 (cross-attn이면 다를 수 있음)
        H, D = self.n_heads, self.head_dim

        # 1) project → reshape → head 차원을 batch 옆으로 끌어올림
        q = self.q_proj(q_in).view(B, Tq, H, D).transpose(1, 2)     # (B, H, Tq, D)
        k = self.k_proj(k_in).view(B, Tk, H, D).transpose(1, 2)     # (B, H, Tk, D)
        v = self.v_proj(v_in).view(B, Tk, H, D).transpose(1, 2)     # (B, H, Tk, D)

        # 2) scaled dot-product: q · k^T / √D  →  (B, H, Tq, Tk) attention matrix
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None:
            # mask=0 인 위치는 -inf 로 만들어 softmax 후 0이 되게 (PAD/미래 위치 차단)
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))

        # 3) value 가중합 → 헤드 다시 합치기 → 출력 projection
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Tq, C)  # (B, Tq, C)
        return self.out_proj(out)


# ───────────────────────────────────────────────────────────────────────
# FFN — position-wise feed-forward network
# ───────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    """
    토큰마다 독립적으로 적용되는 2-layer MLP. d_model → 4×d_model → d_model.
    paper 원본은 ReLU. 모던한 변형은 GELU/SwiGLU/GeGLU 등 사용.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.relu(self.up(x))))


# ───────────────────────────────────────────────────────────────────────
# Encoder Layer — 2 sub-layers: self-attn, FFN
# ───────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Encoder 한 층은 두 개의 sub-layer:
        1. self-attention (양방향 — causal mask 없음, 모든 토큰이 서로를 봄)
        2. FFN

    Pre-LN 구조 사용:
        x = x + sublayer(LN(x))
    Post-LN(원조 paper)보다 학습이 안정적이라 모던 구현은 대부분 Pre-LN.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # sub-layer 1: self-attn  (q=k=v 모두 같은 x)
        n = self.norm1(x)
        x = x + self.attn(n, n, n, src_mask)            # src_mask는 PAD 차단용 (causal 아님)

        # sub-layer 2: FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ───────────────────────────────────────────────────────────────────────
# Decoder Layer — 3 sub-layers: causal self-attn, cross-attn, FFN
# ───────────────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    ★ 인코더-디코더 transformer의 핵심 ★
    Decoder 한 층은 세 개의 sub-layer를 순서대로:
        1. causal self-attention   — decoder가 자기 이전 출력만 보도록 (미래 차단)
        2. cross-attention         — decoder Q × encoder K/V (입력 시퀀스 참조)
        3. FFN

    cross-attention이 "encoder가 제공한 정보를 decoder가 끌어다 쓰는" 통로.
    번역 task로 치면: decoder가 영어 단어를 만들면서 한국어 입력을 쳐다본다.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ── 1. Causal self-attention ─────────────────────────────────────
        # Q=K=V=decoder hidden. tgt_mask가 PAD + 미래 토큰을 동시에 차단.
        n = self.norm1(x)
        x = x + self.self_attn(n, n, n, tgt_mask)

        # ── 2. Cross-attention (★ encoder → decoder 정보 통로) ──────────
        # Q는 decoder, K/V는 encoder output. src_mask는 encoder 쪽 PAD 차단용.
        # 여기서 decoder의 각 위치가 encoder 시퀀스 전체를 자유롭게 참조한다.
        n = self.norm2(x)
        x = x + self.cross_attn(n, enc_out, enc_out, src_mask)

        # ── 3. FFN ───────────────────────────────────────────────────────
        x = x + self.ffn(self.norm3(x))
        return x


# ───────────────────────────────────────────────────────────────────────
# Full Encoder-Decoder Transformer
# ───────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Encoder-decoder transformer (original 2017 paper).

    간소화 사항:
        - src/tgt vocab을 공유 (번역이라면 보통 분리; reverse task는 어차피 동일 어휘)
        - tok_emb과 out_proj weight tying — 메모리/일반화에 유리, paper에서 추천
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # PE는 학습 안 하는 고정 buffer. max_seq_len*2 만큼 넉넉히 만들어 둠.
        # persistent=False → state_dict에 저장 안 됨 (어차피 재생성 가능).
        self.register_buffer(
            "pe", sinusoidal_pe(cfg.max_seq_len * 2, cfg.d_model), persistent=False
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        # Pre-LN이므로 마지막 한 번 더 LN을 걸어줘야 출력 분포가 정규화됨
        self.norm_enc = nn.LayerNorm(cfg.d_model)
        self.norm_dec = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying: 입력 임베딩과 출력 분류기가 같은 행렬을 공유
        self.out_proj.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        # 원조 paper convention: Linear는 Xavier, Embedding은 N(0, 0.02)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        """token id → embedding + positional encoding."""
        # ★ paper trick: embedding을 √d_model 로 scale.
        # 이유: PE는 amplitude 1 짜리 sin/cos 인데, 막 init한 embedding은 분산이 작아서
        #       PE에 묻혀버림. √d_model 로 키워야 둘이 비슷한 크기가 됨.
        x = self.tok_emb(ids) * math.sqrt(self.cfg.d_model)
        x = x + self.pe[: ids.shape[1]].to(x.device)        # 시퀀스 길이만큼 PE 잘라서 더함
        return self.dropout(x)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        """입력 시퀀스를 한 번에 양방향으로 인코딩 → encoder output 반환."""
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.norm_enc(x)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        src_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """현재까지의 tgt + encoder output → 다음 토큰 예측을 위한 hidden."""
        x = self._embed(tgt)
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm_dec(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        학습용 forward (teacher forcing):
            src: 입력 시퀀스
            tgt: 정답 시퀀스의 BOS~끝-1까지 (shifted right)
        """
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return self.out_proj(dec_out)                       # (B, Ttgt, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """
        Greedy decoding: <BOS>로 시작해 한 토큰씩 그리디로 뽑다가 <EOS>가 나오면 종료.

        ★ 효율 포인트: encoder는 입력에 의존하므로 한 번만 돌리고,
                       매 step decoder만 새로 돌린다. (실전에선 KV cache로 더 가속)
        """
        self.eval()
        enc_out = self.encode(src, src_mask)                # encoder 한 번만
        B = src.size(0)
        # decoder 입력을 BOS 한 토큰만으로 시작
        out = torch.full((B, 1), self.cfg.bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            T = out.size(1)
            tgt_mask = make_tgt_mask(out, self.cfg.pad_id, T, src.device)
            dec_out = self.decode(out, enc_out, tgt_mask, src_mask)
            # 마지막 위치의 logits에서 argmax → 다음 토큰
            next_id = self.out_proj(dec_out[:, -1]).argmax(-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
            # 배치 전체가 EOS면 조기 종료
            if (next_id == self.cfg.eos_id).all():
                break
        return out


# ───────────────────────────────────────────────────────────────────────
# Mask utilities
# ───────────────────────────────────────────────────────────────────────
# mask 모양이 (B, 1, 1, T) 인 이유:
#   attention 행렬은 (B, H, Tq, Tk). 거기에 broadcast 되려면 H, Tq 차원이
#   1이거나 같아야 한다. (B, 1, 1, T)로 만들어두면 모든 헤드/모든 query가
#   동일한 PAD mask를 공유하면서 자동으로 broadcast 됨.

def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Encoder용 padding mask. (B, 1, 1, T) — 1=attend, 0=ignore."""
    return (src != pad_id)[:, None, None, :].long()


def make_tgt_mask(tgt: torch.Tensor, pad_id: int, T: int, device) -> torch.Tensor:
    """
    Decoder self-attention용 마스크. PAD 차단 + causal(미래 차단)을 AND 결합.
    Returns: (B, 1, T, T)
    """
    # PAD mask: PAD 토큰 위치를 0으로
    pad = (tgt != pad_id)[:, None, None, :].long()                          # (B, 1, 1, T)
    # Causal mask: 하삼각 행렬, 미래 위치는 0
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.long))[None, None]  # (1, 1, T, T)
    # 둘 중 하나라도 0이면 차단
    return pad & causal


# ───────────────────────────────────────────────────────────────────────
# Synthetic task: sequence reversal
# ───────────────────────────────────────────────────────────────────────

def make_reverse_batch(
    batch_size: int,
    cfg: Config,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    이 학습 kit은 진짜 번역 데이터 없이도 cross-attention의 위력을 보여주려고
    "시퀀스 뒤집기" task를 사용한다.

        src:  [a, b, c, d, EOS, PAD, ...]
        tgt:  [BOS, d, c, b, a, EOS, PAD, ...]   ← 뒤집힌 결과

    이 task는 decoder-only로는 풀기 까다롭다 (전체 입력을 다 본 뒤 거꾸로 출력해야 함).
    encoder-decoder는 cross-attn으로 입력 끝부터 자연스럽게 참조 가능.

    토큰 범위: 3 ~ vocab_size-1 (PAD/BOS/EOS 제외)
    """
    # 시퀀스마다 다른 길이로 (3 ~ max_seq_len/2-1) 골고루 뽑음 → 길이 일반화 학습
    seq_lens = torch.randint(3, cfg.max_seq_len // 2, (batch_size,))
    max_src_len = seq_lens.max().item() + 1                     # +1 for EOS
    max_tgt_len = seq_lens.max().item() + 2                     # +2 for BOS, EOS

    # 미리 PAD로 채워두고 필요한 위치만 덮어쓰기
    src = torch.full((batch_size, max_src_len), cfg.pad_id, dtype=torch.long)
    tgt = torch.full((batch_size, max_tgt_len), cfg.pad_id, dtype=torch.long)

    for i, L in enumerate(seq_lens.tolist()):
        seq = torch.randint(3, cfg.vocab_size, (L,))            # 진짜 어휘에서 무작위
        # src: [seq..., EOS, PAD...]
        src[i, :L] = seq
        src[i, L] = cfg.eos_id
        # tgt: [BOS, seq뒤집기..., EOS, PAD...]
        tgt[i, 0] = cfg.bos_id
        tgt[i, 1 : L + 1] = seq.flip(0)                          # 뒤집기
        tgt[i, L + 1] = cfg.eos_id

    return src.to(device), tgt.to(device)


# ───────────────────────────────────────────────────────────────────────
# LR schedule (warmup + cosine)
# ───────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: Config) -> float:
    """
    Phase 1 (step < warmup_steps): 0 → learning_rate 로 선형 증가
    Phase 2 (이후): cosine 곡선으로 learning_rate → 0 감소
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


# ───────────────────────────────────────────────────────────────────────
# Smoke test — train on reverse task
# ───────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params ({n_params / 1e6:.2f}M)")

    # AdamW + paper-original betas. weight_decay는 LayerNorm bias에도 걸리지만
    # 이 데모에서는 단순화를 위해 그냥 모든 파라미터에 동일하게 적용.
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),                                       # 원조 paper 값 (Adam 기본은 0.999)
    )

    losses: list[float] = []
    t0 = time.time()
    for step in range(cfg.max_steps):
        src, tgt = make_reverse_batch(cfg.batch_size, cfg, device)

        # ★ Teacher forcing의 핵심 한 줄:
        #   decoder 입력은 BOS~정답[T-1] 까지 (tgt[:, :-1])
        #   학습 정답은    정답[1]~정답[T] 까지 (tgt[:, 1:])
        #   → 한 칸씩 어긋나게 해서 "다음 토큰 맞히기"가 됨.
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, cfg.pad_id)
        tgt_mask = make_tgt_mask(tgt_in, cfg.pad_id, tgt_in.size(1), device)

        # forward → cross-entropy. PAD 위치는 ignore_index로 loss에서 제외.
        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            tgt_out.reshape(-1),
            ignore_index=cfg.pad_id,
            label_smoothing=cfg.label_smoothing,
        )

        # 매 step마다 LR을 직접 갱신 (LRScheduler 안 쓰는 minimal 스타일)
        lr = get_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # backward + clip + step 표준 절차
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        losses.append(loss.item())
        if step % cfg.log_interval == 0:
            window = losses[-cfg.log_interval :]
            avg = sum(window) / max(1, len(window))
            print(f"step {step:5d} | lr {lr:.6f} | loss {avg:.4f} | {time.time() - t0:.1f}s")

    # ── Final test: 모델이 reverse를 진짜 학습했나? ───────────────────
    # 학습 때는 teacher forcing(정답을 입력으로 줌)이라 쉬워 보일 수 있음.
    # 진짜 시험은 generate (자기 출력만 보고 한 토큰씩 뽑기).
    print("\n[final] testing reversal generalization...")
    model.eval()
    src, tgt = make_reverse_batch(8, cfg, device)
    src_mask = make_src_mask(src, cfg.pad_id)
    pred = model.generate(src, src_mask, max_len=cfg.max_seq_len)

    correct = 0
    for i in range(src.size(0)):
        # src에서 EOS 전까지가 입력
        src_seq = src[i].tolist()
        eos_idx = src_seq.index(cfg.eos_id) if cfg.eos_id in src_seq else len(src_seq)
        in_seq = src_seq[:eos_idx]
        # tgt와 pred에서 BOS 다음 ~ EOS 전까지가 출력
        tgt_seq = tgt[i].tolist()
        pred_seq = pred[i].tolist()
        tgt_out = tgt_seq[1 : tgt_seq.index(cfg.eos_id) if cfg.eos_id in tgt_seq[1:] else len(tgt_seq)]
        pred_after_bos = pred_seq[1:] if pred_seq[0] == cfg.bos_id else pred_seq
        pred_out = pred_after_bos[: pred_after_bos.index(cfg.eos_id)] if cfg.eos_id in pred_after_bos else pred_after_bos

        ok = pred_out == tgt_out
        if ok:
            correct += 1
        marker = "✓" if ok else "✗"
        print(f"  {marker} src={in_seq}  →  pred={pred_out}  (expected: {tgt_out})")

    print(f"\n[accuracy] {correct}/{src.size(0)} sequences reversed correctly")


if __name__ == "__main__":
    smoke_test()
