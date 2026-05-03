"""
sonlm — Vanilla decoder-only transformer (학습 reference).

Architecture: multi-head attention, ReLU FFN, LayerNorm (Pre-LN),
learned positional embeddings, weight-tied LM head.
No GQA, no SwiGLU, no parallel residual, no RoPE — 모던 기법은 의도적으로 제외.
이 단순함이 출발점이고, 모던 기법들은 experiments/ 폴더에서 ablation으로 비교.

────────────────────────────────────────────────────────────────────────────
[전체 흐름 한 눈에 보기]

  입력 토큰 idx (B, T)                       ← 정수 ID들
        │
        ▼  tok_emb + pos_emb
  x (B, T, d_model)                         ← 벡터로 변환됨
        │
        ▼  Block × n_layers
  x (B, T, d_model)                         ← 각 블록: Attention → FFN
        │                                       (Pre-LN + residual)
        ▼  final LayerNorm + lm_head
  logits (B, T, vocab_size)                 ← 각 위치에서 다음 토큰 분포

기호:
  B = batch size       (예: 32)
  T = sequence length  (예: 128, 토큰 개수)
  C = d_model          (예: 384, 임베딩 차원)
  H = n_heads          (예: 6)
  D = head_dim = C/H   (예: 64)
────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SonLMConfig


class Attention(nn.Module):  # multi-head self-attention
    """
    Self-Attention: 각 토큰이 "다른 토큰들을 얼마나 볼지" 결정해서 정보를 섞음.
    Multi-head: 한 번에 하지 않고 H개의 head로 쪼개서 병렬로 attention 수행.
                각 head는 서로 다른 관점(예: 문법, 의미 등)을 학습할 수 있음.
    """
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads                       # H: head 개수
        self.head_dim = config.d_model // config.n_heads    # D: head 하나의 차원 (C / H)

        # Q, K, V를 각각 만드는 대신 한 번에 3*C 차원으로 만들고 나중에 쪼갬 (속도 ↑)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        # head들을 합친 뒤 다시 한 번 섞어주는 출력 projection
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        # x: (B, T, C)
        B, T, C = x.shape

        # ── 1) Q, K, V 만들기 ────────────────────────────────────────────────
        # qkv(x): (B, T, 3*C)
        # reshape:   (B, T, 3, H, D)   ← 3개(QKV) × H개 head × D차원으로 분해
        # permute:   (3, B, H, T, D)   ← 맨 앞 3을 0번으로 빼서 q,k,v 슬라이싱 쉽게
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # 각각 (B, H, T, D)

        # ── 2) Attention score 계산 ─────────────────────────────────────────
        # k.transpose(-2,-1): (B, H, D, T)
        # q @ k^T:            (B, H, T, T)   ← "토큰 i가 토큰 j를 얼마나 볼지" 점수표
        # √D로 나누는 이유: 차원이 커질수록 내적값이 커져서 softmax가 너무 sharp해지는 걸 방지
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ── 3) Causal mask 적용 ──────────────────────────────────────────────
        # 미래 토큰을 보면 안 됨 (autoregressive). 미래 위치를 -inf로 채우면
        # softmax 후 0이 되어 무시됨.
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # softmax로 확률화 → dropout (attention dropout)
        attn = self.dropout(F.softmax(attn, dim=-1))   # (B, H, T, T)

        # ── 4) Value들 가중합 + head 합치기 ──────────────────────────────────
        # attn @ v:               (B, H, T, D)
        # transpose(1,2):         (B, T, H, D)         ← head를 다시 옆으로
        # contiguous().view:      (B, T, C)            ← H, D를 합쳐 C 복원
        # self.out(...):          (B, T, C)            ← 마지막으로 한 번 더 섞기
        return self.out((attn @ v).transpose(1, 2).contiguous().view(B, T, C))


class FFN(nn.Module):
    """
    Feed-Forward Network: position-wise MLP.
    Attention이 토큰 '사이' 정보를 섞었다면, FFN은 각 토큰 '내부'에서 비선형 변환.
    구조: C → ffn_hidden(보통 2~4배) → C, 사이에 ReLU.
    """
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_hidden)    # 차원 확장
        self.down = nn.Linear(config.ffn_hidden, config.d_model)  # 다시 축소
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (B, T, C) → up: (B, T, ffn_hidden) → ReLU → down: (B, T, C)
        return self.dropout(self.down(F.relu(self.up(x))))


class Block(nn.Module):
    """
    Transformer 블록 하나 = Attention + FFN, 둘 다 Pre-LN + residual.

    Pre-LN 패턴:  x = x + sublayer(LayerNorm(x))
       - LayerNorm을 sublayer '앞'에 둠 (Post-LN보다 학습 안정적)
       - residual(x +)은 gradient가 깊은 모델에서도 잘 흐르게 해줌
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)   # attention sub-layer
        x = x + self.ffn(self.norm2(x))          # FFN sub-layer
        return x


class SonLM(nn.Module):
    """
    전체 모델: 임베딩 → Block × N → 최종 LayerNorm → lm_head(logits).
    """
    def __init__(self, config: SonLMConfig):
        super().__init__()
        self.config = config

        # 토큰 ID → 벡터.  weight shape: (vocab_size, d_model)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # 위치(0,1,2,...) → 벡터. learned positional embedding (RoPE 같은 거 안 씀).
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # N개의 transformer 블록을 쌓음
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])

        # 마지막 블록 출력에 한 번 더 LayerNorm (Pre-LN 모델의 표준)
        self.norm = nn.LayerNorm(config.d_model)

        # d_model → vocab_size: 각 위치에서 "다음 토큰" 분포의 점수(logit)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: 입력 임베딩과 출력 projection의 weight를 공유.
        # 파라미터 수 절약 + 일반적으로 성능에도 유리.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # GPT 계열 표준 초기화: N(0, 0.02)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, T)  ← 정수 토큰 ID
        B, T = idx.shape

        # ── 1) 임베딩 ───────────────────────────────────────────────────────
        # tok_emb(idx):   (B, T, C)        토큰 → 벡터
        # pos_emb(pos):   (T, C) → broadcast로 (B,T,C)에 더해짐    위치 정보 주입
        pos = torch.arange(T, device=idx.device)             # (T,)  [0,1,2,...,T-1]
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)) # (B, T, C)

        # ── 2) Causal mask 만들기 ───────────────────────────────────────────
        # tril(ones(T,T)) → 하삼각 행렬 (1=볼 수 있음, 0=가림)
        # 예 T=4:
        #   1 0 0 0      (위치 0은 자기 자신만)
        #   1 1 0 0      (위치 1은 0,1까지)
        #   1 1 1 0      (위치 2는 0,1,2까지)
        #   1 1 1 1      (위치 3은 전부)
        # unsqueeze 두 번 → (1, 1, T, T)로 만들어서 (B, H, T, T) attn에 broadcast
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

        # ── 3) Block들 통과 ──────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, mask)                               # (B, T, C) 유지

        # ── 4) 최종 LayerNorm + lm_head로 logits 산출 ──────────────────────
        logits = self.lm_head(self.norm(x))                  # (B, T, vocab_size)

        # ── 5) 학습 모드면 loss도 같이 계산 ─────────────────────────────────
        loss = None
        if targets is not None:
            # cross_entropy는 (N, vocab) vs (N,) 형식을 원하므로 평탄화.
            # ignore_index=0: pad 토큰(=0)이 있는 위치는 loss에서 제외.
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),    # (B*T, vocab)
                targets.view(-1),                            # (B*T,)
                ignore_index=0,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.7, top_k=50, **kwargs):
        """
        Autoregressive 생성: 한 토큰씩 만들어서 idx 뒤에 붙임.
          temperature: logit을 나누는 값. 작을수록 확신 있게(=결정적), 클수록 다양하게.
          top_k:       상위 k개 후보만 남기고 나머지 확률을 0으로. 헛소리 방지.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # context가 max_seq_len 넘으면 뒤쪽만 자름 (KV cache 안 쓰는 단순 구현)
            idx_cond = idx[:, -self.config.max_seq_len:]

            # 전체 시퀀스에 대해 logits을 뽑지만 우리가 필요한 건 마지막 위치 뿐
            logits, _ = self(idx_cond)                # (B, T, vocab)
            logits = logits[:, -1, :] / temperature   # (B, vocab)  ← 마지막 토큰 logit만

            # top-k filtering: 상위 k개 외에는 -inf로 보내 softmax 후 0이 되게
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))   # v: (B, k)
                logits[logits < v[:, [-1]]] = float("-inf")              # k번째 미만 차단

            # 확률화 후 샘플링 (argmax 아니고 multinomial → 다양성 유지)
            probs = F.softmax(logits, dim=-1)                            # (B, vocab)
            next_id = torch.multinomial(probs, num_samples=1)            # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)                       # (B, T+1)

            # EOS면 종료 (주의: batch_size>1이면 첫 샘플 기준으로만 멈춤)
            if next_id.item() == self.config.eos_id:
                break
        return idx, []

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return total, 0

    def param_summary(self):
        total, _ = self.param_count()
        return f"SonLM: {total:,} params ({total/1e6:.1f}M)"
