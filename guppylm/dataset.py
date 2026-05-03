"""
GuppyLM dataset loading.

────────────────────────────────────────────────────────────────────────────
[이 파일이 하는 일 한 눈에 보기]

  data/train.jsonl                         ← 한 줄에 한 샘플 (JSON Lines)
  {"text": "hi guppy <sep> hello..."}
        │
        ▼  Tokenizer (BPE)
  ids = [152, 891, 23, 7, ...]             ← 정수 ID 리스트
        │
        ▼  __getitem__: 한 칸 shift
  x = ids[:-1]   y = ids[1:]               ← x는 입력, y는 정답
        │
        ▼  collate_fn: 길이 맞춰 padding
  (B, T_max)                               ← 배치 텐서
        │
        ▼  DataLoader가 학습 루프로 전달
  model(x, targets=y)                      ← model.py의 forward에 들어감

핵심: language modeling = "다음 토큰 맞추기".
      x[i]를 보고 y[i] (= x[i+1])을 예측하도록 학습.
────────────────────────────────────────────────────────────────────────────
"""

import json

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


class GuppyDataset(Dataset):
    """
    JSONL 파일을 읽어서 토큰화한 ID 리스트들을 메모리에 올려둠.
    PyTorch Dataset이라 __len__과 __getitem__만 구현하면 DataLoader가 알아서 씀.
    """
    def __init__(self, path: str, tokenizer_path: str, max_len: int = 512):
        # 학습된 BPE 토크나이저 로드 (prepare_data.py에서 만들어둔 파일)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.samples = []   # 각 원소는 정수 ID 리스트 (샘플 하나)

        # JSONL 한 줄 = 한 샘플. 모든 샘플을 미리 토큰화해서 메모리에 올림.
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                # 문자열 → 정수 ID 리스트.   예: "hi guppy" → [152, 891, ...]
                ids = self.tokenizer.encode(data["text"]).ids

                # max_len(예: 128) 넘으면 잘라냄. 모델의 max_seq_len과 맞춰야 함.
                if len(ids) > max_len:
                    ids = ids[:max_len]

                # 너무 짧은 샘플(0~1 토큰)은 x/y 만들 수 없으니 버림.
                # __getitem__의 x = ids[:-1], y = ids[1:] 가 비어버리는 걸 방지.
                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        한 샘플 → (x, y) 쌍 반환.

        ⭐ 핵심: 한 칸 shift!
          ids = [A, B, C, D, E]
          x   = [A, B, C, D]     ← 모델이 보는 입력
          y   = [B, C, D, E]     ← 각 위치에서 맞춰야 할 정답
                                    위치 0에서 A→B 예측,
                                    위치 1에서 B→C 예측, ... (causal mask 덕분)

        한 번의 forward로 시퀀스 길이만큼의 예측 문제를 동시에 학습.
        """
        ids = self.samples[idx]
        x = ids[:-1]                                    # 마지막 토큰 제외
        y = ids[1:]                                     # 첫 토큰 제외 (한 칸 밀림)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    """
    DataLoader가 한 배치를 만들 때 호출하는 함수.

    문제: 샘플마다 길이가 다름 (어떤 건 20 토큰, 어떤 건 80 토큰).
          텐서로 쌓으려면 같은 길이여야 함.
    해결: 가장 긴 샘플 길이에 맞춰 pad_id(=0)로 뒤를 채움.

    참고: model.py의 cross_entropy에서 ignore_index=0을 줘서
          pad 위치는 loss 계산에서 빠짐.
    """
    # batch: [(x_0, y_0), (x_1, y_1), ...]   ← __getitem__이 만든 튜플들의 리스트
    # zip(*batch) → xs = (x_0, x_1, ...),  ys = (y_0, y_1, ...)
    xs, ys = zip(*batch)

    # 이 배치의 최대 길이 (배치마다 다름 → "dynamic padding")
    max_len = max(len(x) for x in xs)

    # 0으로 가득 찬 (B, max_len) 텐서를 만들고 앞부분만 실제 토큰으로 채움
    padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :len(x)] = x       # 예: [A,B,C,0,0,0]
        padded_y[i, :len(y)] = y       # 예: [B,C,D,0,0,0]
    return padded_x, padded_y           # 둘 다 (B, max_len)


def get_dataloader(path, tokenizer_path, max_len=512, batch_size=32, shuffle=True):
    """
    DataLoader 만드는 헬퍼.
    학습 루프에서 `for x, y in loader:` 로 돌리면
    (B, T_max) 텐서가 매 step마다 흘러나옴.
    """
    dataset = GuppyDataset(path, tokenizer_path, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,            # epoch마다 순서 섞기 (학습용은 True, 평가용은 False)
        collate_fn=collate_fn,      # 위에서 만든 padding 함수
        num_workers=0,              # 0 = 메인 프로세스에서 데이터 로딩 (작은 데이터셋이라 충분)
        pin_memory=True,            # GPU 전송 속도 향상 (CUDA 사용 시)
    )
