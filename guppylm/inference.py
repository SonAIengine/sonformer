"""
GuppyLM inference — simple chat.

────────────────────────────────────────────────────────────────────────────
[추론 흐름 한 눈에 보기]

  사용자 입력 messages = [{"role": "user", "content": "hi guppy"}]
        │
        ▼  _format_prompt: ChatML 형식으로 포장
  prompt = "<|im_start|>user\nhi guppy<|im_end|>\n<|im_start|>assistant\n"
        │
        ▼  tokenizer.encode
  input_ids = [1, 152, 891, 2, 1, 304, ...]            ← 정수 ID
        │
        ▼  model.generate (autoregressive)             ← model.py 참고
  output_ids = [1, 152, 891, 2, 1, 304, ..., <new_tokens>]
        │
        ▼  prompt 부분 잘라내고 decode
  "hello. the water is nice today.<|im_end|>"
        │
        ▼  <|im_end|> 이후 제거
  "hello. the water is nice today."

핵심: 학습은 "다음 토큰 예측"의 반복, 추론도 똑같이 "다음 토큰 예측"의 반복.
      차이는 단지 "정답 비교 안 하고, 예측한 걸 그대로 다음 입력에 이어붙임"뿐.

ChatML 포맷:
  <|im_start|>{role}\n{content}<|im_end|>
  → 모델은 학습 때 이 패턴을 봤으므로, "<|im_start|>assistant\n" 뒤에
    이어서 답변을 생성하도록 자연스럽게 유도됨.
────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import time
import uuid

import torch
from tokenizers import Tokenizer

from .config import GuppyConfig
from .model import GuppyLM


class GuppyInference:
    """학습된 체크포인트를 로드해서 chat 응답을 생성하는 엔진."""

    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # ── 1) 체크포인트 로드 ──────────────────────────────────────────────
        # train.py가 저장한 dict: {"step", "model_state_dict", "config", ...}
        # weights_only=False: config 등 dict도 같이 로드 (PyTorch 2.6+ 기본이 True라 명시)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # config.json은 모델 파일과 같은 폴더에 있다고 가정 (train.py가 그렇게 저장)
        config_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        config_path = os.path.join(config_dir, "config.json")

        # ── 2) state_dict 추출 ──────────────────────────────────────────────
        # 표준 형식이면 dict 안에 들어있고, 옛날 형식이면 ckpt 자체가 state_dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # ── 3) Config 복원 ──────────────────────────────────────────────────
        # 우선순위: 외부 config.json > ckpt 내부 config > 기본값.
        # HuggingFace 표준 키 이름과 우리 이름 둘 다 지원 (HF에 올린 모델도 로드 가능).
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.config = GuppyConfig(
                vocab_size=cfg.get("vocab_size", 4096),
                max_seq_len=cfg.get("max_position_embeddings", cfg.get("max_seq_len", 128)),
                d_model=cfg.get("hidden_size", cfg.get("d_model", 384)),
                n_layers=cfg.get("num_hidden_layers", cfg.get("n_layers", 6)),
                n_heads=cfg.get("num_attention_heads", cfg.get("n_heads", 6)),
                ffn_hidden=cfg.get("intermediate_size", cfg.get("ffn_hidden", 768)),
                dropout=cfg.get("hidden_dropout_prob", cfg.get("dropout", 0.1)),
                pad_id=cfg.get("pad_token_id", cfg.get("pad_id", 0)),
                bos_id=cfg.get("bos_token_id", cfg.get("bos_id", 1)),
                eos_id=cfg.get("eos_token_id", cfg.get("eos_id", 2)),
            )
        elif isinstance(ckpt, dict) and "config" in ckpt:
            # 학습 당시의 config가 ckpt에 같이 들어있는 경우
            valid_fields = {f.name for f in GuppyConfig.__dataclass_fields__.values()}
            self.config = GuppyConfig(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
        else:
            print("Warning: No config found, using defaults")
            self.config = GuppyConfig()

        # ── 4) 모델 생성 + weight 로드 ──────────────────────────────────────
        self.model = GuppyLM(self.config).to(self.device)
        # 모델에 실제로 존재하는 키만 골라서 로드 (옛 체크포인트 호환).
        # 새 모델에 없는 키가 있으면 strict=True에서 에러 나는 걸 방지.
        filtered = {k: v for k, v in state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(filtered)
        self.model.eval()                           # dropout 끔 (추론 모드)

        total, _ = self.model.param_count()
        print(f"GuppyLM loaded: {total/1e6:.1f}M params")

    def chat_completion(self, messages, temperature=0.7, max_tokens=64,
                        top_k=50, **kwargs):
        """
        messages를 받아서 응답 텍스트를 생성.
        반환 형식이 OpenAI Chat Completions API와 비슷한 모양 (호환성).

        파라미터:
          temperature: 작을수록 결정적, 클수록 다양함 (model.py의 generate 참고)
          max_tokens:  생성할 최대 토큰 수
          top_k:       매 step마다 상위 k개 후보로 제한
        """
        # ── 1) 프롬프트 만들기 ─────────────────────────────────────────────
        prompt = self._format_prompt(messages)
        input_ids = self.tokenizer.encode(prompt).ids
        prompt_tokens = len(input_ids)              # 나중에 새로 생성된 부분만 자르려고 기억

        # 배치 차원 추가: (T,) → (1, T)   ← 모델은 (B, T) 형태를 요구
        input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # ── 2) Autoregressive 생성 ─────────────────────────────────────────
        # model.generate가 input_t 뒤에 새 토큰을 붙여서 돌려줌.
        # output_t shape: (1, prompt_tokens + 생성된_토큰수)
        output_t, _ = self.model.generate(input_t, max_tokens, temperature, top_k)

        # ── 3) 새로 생성된 부분만 잘라내서 디코딩 ──────────────────────────
        # [prompt_tokens:] → 사용자 입력 부분은 빼고 모델이 생성한 부분만
        output_text = self.tokenizer.decode(output_t[0].tolist()[prompt_tokens:])

        # ── 4) 후처리: 특수 토큰 누수 차단 ─────────────────────────────────
        # 모델이 종종 <|im_end|> 뒤에 다음 턴까지 이어서 생성하는 경우가 있음.
        # 첫 <|im_end|> 이후는 잘라서 응답에 다음 user 차례가 섞이지 않게 함.
        if "<|im_end|>" in output_text:
            output_text = output_text.split("<|im_end|>")[0]
        if "<|im_start|>" in output_text:
            output_text = output_text.split("<|im_start|>")[0]
        resp_text = output_text.strip()

        # OpenAI 호환 형식으로 포장
        return {
            "choices": [{
                "message": {"role": "assistant", "content": resp_text},
            }],
        }

    def _format_prompt(self, messages):
        """
        messages 리스트를 ChatML 형식 문자열로 변환.

        예: [{"role": "user", "content": "hi"}]
        →  "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
                                              ↑
                                          이렇게 끝내면 모델이
                                          assistant 답변을 이어서 생성하도록 유도됨
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        # 마지막에 assistant 시작 토큰만 두고 끝 → 모델이 이어서 답변 생성
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def main():
    """CLI 엔트리포인트: `python -m guppylm chat`이 결국 이걸 호출."""
    import argparse
    p = argparse.ArgumentParser(description="Chat with Guppy")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", "-p", help="Single prompt mode: ask one question and exit")
    args = p.parse_args()

    # 모델 로드 (한 번만 — 매 입력마다 다시 로드하면 느려짐)
    engine = GuppyInference(args.checkpoint, args.tokenizer, args.device)

    # ── 단발 모드: --prompt 인자 있으면 한 번 답하고 종료 ──────────────────
    if args.prompt:
        result = engine.chat_completion([{"role": "user", "content": args.prompt}])
        print(result["choices"][0]["message"]["content"])
        return

    # ── 인터랙티브 모드: 사용자 입력을 계속 받음 ─────────────────────────
    # 주의: 매 턴마다 새 messages 리스트를 만들어서 단일 턴으로 동작.
    #       대화 히스토리를 누적하지 않음 → 128 토큰 제한 안에서 안정적.
    print("\nGuppy Chat (type 'quit' to exit)")
    while True:
        inp = input("\nYou> ").strip()
        if inp.lower() in ("quit", "exit", "q"):
            break
        result = engine.chat_completion([{"role": "user", "content": inp}])
        msg = result["choices"][0]["message"]
        if msg.get("content"):
            print(f"Guppy> {msg['content']}")


if __name__ == "__main__":
    main()
