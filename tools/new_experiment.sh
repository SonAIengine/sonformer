#!/usr/bin/env bash
# 새 ablation 실험 폴더 스폰.
#
# Usage:
#   bash tools/new_experiment.sh 02b-swiglu
#   bash tools/new_experiment.sh 02c-rmsnorm "LayerNorm을 RMSNorm으로 교체"
#
# 동작:
#   1. experiments/00-baseline/ → experiments/<name>/ 복사
#   2. config.yaml의 experiment.name + description 갱신
#   3. README.md를 가설 작성 템플릿으로 초기화
#   4. results/ 비움
#   5. 다음 단계 안내

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash tools/new_experiment.sh <name> [description]"
  echo "Example: bash tools/new_experiment.sh 02b-swiglu \"FFN: ReLU → SwiGLU\""
  exit 1
fi

NAME="$1"
DESC="${2:-TODO}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/experiments/00-baseline"
DST="$REPO_ROOT/experiments/$NAME"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: $SRC not found"
  exit 1
fi

if [[ -d "$DST" ]]; then
  echo "ERROR: $DST already exists"
  exit 1
fi

cp -r "$SRC" "$DST"
rm -rf "$DST/results"/* 2>/dev/null || true
mkdir -p "$DST/results"
touch "$DST/results/.gitkeep"

# config.yaml의 experiment.name + description 갱신
python3 - <<EOF
import yaml
from pathlib import Path

p = Path("$DST/config.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["experiment"]["name"] = "$NAME"
cfg["experiment"]["description"] = "$DESC"
p.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True))
print(f"  updated config.yaml: name={cfg['experiment']['name']}")
EOF

# README.md를 가설 템플릿으로 초기화
cat > "$DST/README.md" <<HEREDOC
# $NAME — $DESC

> _$DESC_

---

## 1. 가설

| Metric | 예상 | 실제 |
|---|---|---|
| In-distribution PPL | _ | _TBD_ |
| Wallclock | _ | _TBD_ |
| Params | _ | _TBD_ |

(예측 채우고, 학습 후 실제 컬럼 채움)

---

## 2. Baseline 대비 변경점

### 2-1. \`model.py\`

\`\`\`python
# baseline:
...

# this experiment:
...
\`\`\`

### 2-2. \`config.yaml\` (있다면)

\`\`\`yaml
# 추가/변경된 키
\`\`\`

---

## 3. 데이터셋

(baseline과 동일하게 유지하는 것이 비교 공정성에 중요)

---

## 4. 학습 결과

### In-distribution

| Metric | 00-baseline | $NAME | Δ |
|---|---|---|---|
| Params | _ | _ | _ |
| Final eval PPL | _ | _ | _ |
| Wallclock | _ | _ | _ |
| Peak VRAM | _ | _ | _ |

### 정성 샘플

\`\`\`
prompt: ...
gen:    ...
\`\`\`

---

## 5. 결론

(실측 후 작성)

---

## 6. 한 줄 정리

> ...
HEREDOC

echo ""
echo "✓ Created: experiments/$NAME"
echo ""
echo "Next steps:"
echo "  1. Edit experiments/$NAME/model.py — apply your one architectural change"
echo "  2. Update config.yaml if new hyperparameters needed"
echo "  3. Fill in 'Hypothesis' section of experiments/$NAME/README.md"
echo "  4. Run:   cd experiments/$NAME && SONFORMER_SKIP_DEPS=1 bash run.sh"
echo "  5. Fill results into README.md"
echo ""
