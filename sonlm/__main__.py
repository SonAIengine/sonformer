"""Entry point for: python -m sonlm

학습용 reference 패키지 (sonformer의 "교과서").
실제 ablation 실험은 experiments/ 폴더에서 진행.
"""

import os
import sys

CHECKPOINT_PATH = "checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"


def main():
    if len(sys.argv) < 2:
        print("sonlm — vanilla decoder-only transformer (학습 reference)")
        print()
        print("Usage:")
        print("  python -m sonlm prepare      Generate synthetic data & train BPE tokenizer")
        print("  python -m sonlm train        Train the model end-to-end")
        print("  python -m sonlm chat         Chat with the trained model")
        print()
        print("본격 ablation은 experiments/ 폴더의 자기완결 실험에서 진행.")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]

    if cmd == "prepare":
        from .prepare_data import prepare
        prepare()

    elif cmd == "train":
        from .train import train
        train()

    elif cmd == "chat":
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"Model not found at {CHECKPOINT_PATH}.")
            print("Train first:\n")
            print("  python -m sonlm prepare")
            print("  python -m sonlm train")
            return

        from .inference import main as inference_main
        inference_main()

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m sonlm' for usage.")


main()
