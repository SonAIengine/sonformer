# 자기 참조 / 순환 참조 타입
from __future__ import annotations 

import math
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 데코레이터이고 함수/클래스를 다른 함수로 감싸서 변형하는 파이썬 문법.
# 본질: 그냥 함수 호출
@dataclass
class Config:
  # Vocab + special tokens
  vocab_size: int = 32
  pad_id: int = 0 # 길이 맞추기용 빈칸. attention/loss에서 무시됨
  bos_id: int = 1 # 출력 시작 신호 decoder의 generate 첫 입력
  eos_id: int = 2 # 출력 끝 신호. generate 루프 종료 조건
  
  # Model
  d_model: int = 128
  n_head: int = 4
  n_layers: int = 3
  ffn_hidden: int = 512
  max_seq_len: int = 16
  dropout: float = 0.1
  
  # Training 
  batch_size: int = 64
  learning_rate: float = 3e-4
  warmup_steps: int = 100
  max_steps: int = 1500
  weight_decay: float = 0.01
  grad_clip: float = 1.0
  label_smoothing: float = 0.1
  logl_smotting float = 0.1
  log_interval: int = 100
  seed: int = 42
  
# Sinusoidal Positional Encoding (paper-original)

