"""sonlm — vanilla decoder-only transformer for transformer/LM 학습 reference.

이 패키지는 sonformer 프로젝트의 "교과서" 역할 (한국어 학습 주석 포함).
실제 ablation 실험은 experiments/ 폴더의 자기완결 셋업에서 진행.
"""

from .config import SonLMConfig, TrainConfig
from .model import SonLM

__version__ = "0.1.0"
