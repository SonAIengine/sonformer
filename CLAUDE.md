# CLAUDE.md — sonformer 협업 가이드

> 이 파일은 Claude Code 세션이 자동으로 로드합니다.
> 사용자(SonAIengine)와 함께 transformer/LLM 학습 ablation을 진행하는 환경입니다.

---

## 프로젝트 한 줄 정의

**sonformer** = vanilla decoder-only transformer를 baseline으로 두고, 각 모던 기법을 한 가지씩 ablation해서 효과를 손으로 측정하는 학습 kit.

원본: [arman-bd/guppylm](https://github.com/arman-bd/guppylm) (MIT). 본 저장소는 `SonAIengine/sonformer`.

---

## 박제(frozen) vs 진화(evolving) 영역

| 영역 | 수정 가능? | 설명 |
|---|---|---|
| `sonlm/` | ❌ **절대 금지** | vanilla decoder-only LM (한국어 학습 주석). "교과서" 역할로 박제. |
| `experiments/00-baseline/` | ❌ **절대 금지** | 모든 ablation의 비교 기준점. 첫 측정 후 봉인. |
| `experiments/<NN-name>/` | ✅ 자유 | 각 ablation 실험 폴더. 새로 만들거나 결과 채우기. |
| `practice/<topic>/` | ⚠️ **사용자만** | 사용자가 직접 손코딩하는 학습 공간. **Claude는 새 코드를 채워주지 말 것** — 가이드/힌트/디버깅만. 단, 사용자가 명시적으로 요청한 정답지(`solution.py` 등)는 예외로 한 번 작성 후 frozen. |
| `shared/` | ✅ 신중하게 | 모든 실험이 공유하는 인프라. 변경 시 모든 실험에 영향. |
| `README.md`, `experiments/README.md` | ✅ | 새 ablation 결과 표 갱신 |
| `CLAUDE.md` (이 파일) | ✅ | 협업 방식 합의 변경 시 업데이트 |

**baseline이나 sonlm을 만지려는 충동이 들면 멈추고 사용자에게 먼저 확인.**

---

## 사용자 협업 모드 (이게 이 파일의 핵심)

사용자는 대학원 과정에서 transformer/LLM 아키텍처와 학습 파이프라인을 **직접 구현하며** 공부합니다. 다음 발화 패턴을 신호로 인식:

### 발화 → 폴더 매핑 (practice vs experiments 구분 핵심)

같은 기법(예: RoPE)이 두 폴더 모두에 등장 가능. **목적에 따라** 어디로 갈지 결정:

| 사용자 발화 (예시) | 어느 트랙? | 즉시 해야 할 일 |
|---|---|---|
| "**X 직접 손코딩해보고 싶어**" / "**X 어떻게 동작해? 짜보면서 이해하자**" | 🖐️ practice | `practice/<NN-X>/` 폴더 + README 가이드. **코드는 사용자가 직접**, Claude는 README/힌트/디버깅만. synthetic toy task로 동작 확인이 목표. |
| "**X ablation 돌려서 baseline 대비 PPL 보자**" / "**X 효과 측정해보자**" | 🔬 experiments | `experiments/<NN-X>/` 스폰 → 가설부터 같이 작성 (예상 PPL/속도/VRAM). 진짜 데이터(Shakespeare/TinyStories)로 정량 측정. |
| "**X 공부하고 싶어**" (모호) | 둘 다 가능 | "구조 이해 먼저(practice)? 효과 측정(experiments)? 어느 쪽?" 한 번 확인. **권장 동선: practice 손코딩 → experiments ablation**. |
| "**왜 PPL 차이가 이렇게 나?**" | 🔬 experiments | 분석 (loss curve, attention map 등) |
| "**이 논문 한 번 따라가보자**" | 보통 둘 다 | 논문 → 구조는 practice, 측정은 experiments로 매핑 |
| "**한국어 데이터로 돌려보고싶어**" | 🔬 experiments | Phase 10, `experiments/10a-korean-data/` |

핵심: **답을 한 번에 던지지 않는다.** 단계 나눠서, "왜 그렇게 하는지"를 설명하면서 같이 코드 작성.

---

## 새 ablation 표준 워크플로

사용자가 "X 공부하고 싶어"라고 하면:

```
1. 무엇을 측정할지 합의
   - 어떤 컴포넌트를 바꾸나? (Attention? FFN? Norm?)
   - 핵심 metric은? (PPL / wallclock / VRAM / extrapolation)
   - 가설은? (예: "SwiGLU는 ReLU 대비 PPL 5% 개선")

2. 폴더 스폰
   bash tools/new_experiment.sh 02b-swiglu

3. README.md 가설 먼저 작성
   - "왜 이 변경을 해보는가"
   - "예상 결과 vs 실제" 표 (실제는 비워둠)

4. model.py 한 줄씩 수정
   - 사용자에게 변경 의도 설명
   - diff를 명확히 보여주기 (어떤 라인이 어떻게 바뀌는지)
   - 수학적 직관 + 구현상 주의점

5. config.yaml 조정 (필요 시)
   - 새 하이퍼파라미터 추가 (예: rope_base, gqa_n_kv_heads)
   - 다른 모든 값은 baseline과 동일 유지 (비교 공정성)

6. 실행
   cd experiments/02b-swiglu && bash run.sh
   - 짧은 sanity 먼저 (Shakespeare 1500 step)
   - 잘 되면 본격 (TinyStories 5000 step)

7. 결과를 README의 표에 채움
   - eval PPL, wallclock, VRAM, params delta
   - 정성적 샘플 한두 개

8. commit
```

---

## 코딩 컨벤션

- **코드 자체**: 영문 docstring 위주 (grep 친화). 한국어 주석은 정말 비직관적인 곳에만 짧게.
- **README/노트**: 한국어. "왜 이렇게 했나"의 의도와 한계를 명확히.
- **commit 메시지**: 한국어 OK. 본문은 변경의 *왜*를 짧게. 마지막에 `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` 붙이기.
- **클래스명**: 패키지 `sonlm`은 `SonLM` (학습 reference). 각 실험 폴더의 model.py는 `Sonformer` (ablation 변형들의 공통 클래스명).
- **dataclass Config**: 모든 모델 파라미터를 한 곳에. YAML → Config → 모델 흐름.
- **파라미터 변경**: shared/ 인프라 함수 시그니처 바꿀 땐 모든 실험에 영향이 가니 신중하게.

---

## 파일 구조 빠른 참조

```
sonformer/
├── CLAUDE.md                  ← 이 파일
├── README.md                  ← 프로젝트 소개 + 8-Phase 로드맵
├── sonlm/                     ❌ frozen 교과서 (한국어 주석, class SonLM)
├── shared/                    ⚠️  공통 인프라 (조심히 변경)
│   ├── datasets.py              load_dataset("shakespeare"|"tinystories"|"wikitext-103"|"pg19")
│   ├── tokenizers.py            get_tokenizer(dataset, vocab) — BPE 자동 학습/캐시
│   ├── eval.py                  perplexity, extrapolation_perplexity, generate_samples
│   └── logging.py               CSVLogger, save_json, save_samples
├── experiments/
│   ├── README.md                실험 인덱스 + 결과 요약
│   ├── 00-baseline/             ❌ frozen 기준점 (PPL 37.47 on Shakespeare)
│   └── 02a-rope/                ✅ 첫 ablation (PPL 28.47 on Shakespeare)
├── data/                      📦 다운로드 캐시 (.gitignore, 토크나이저는 commit)
├── practice/                  🖐️ 사용자 손코딩 공간 (Claude는 가이드만, 코드는 채우지 말 것)
│   ├── README.md                  practice 인덱스 + 8-tier 커리큘럼 (Tier 0~7)
│   ├── 00-tensor-autograd/        tensor + autograd 손풀이 (Tier 0 기초)
│   ├── 01-mlp-from-scratch/       MLP on two-moons (Tier 0 기초)
│   ├── 02-cnn-from-scratch/       1D TextCNN (Tier 1 pre-transformer)
│   ├── 03-rnn-from-scratch/       Vanilla Elman RNN (Tier 1)
│   ├── 04-lstm-from-scratch/      4-gate LSTM (Tier 1)
│   ├── 05-sonlm-from-scratch/     decoder-only transformer (Tier 2, GPT 계열)
│   ├── 06-encoder-decoder-from-scratch/  원조 2017 encoder-decoder (Tier 2)
│   └── 07-bert-from-scratch/      encoder-only + MLM (Tier 2, BERT)
└── tools/
    ├── new_experiment.sh        새 ablation 폴더 스폰
    └── compare.py               여러 실험 결과 비교 표 출력
```

---

## 표준 명령어

```bash
# 새 실험 시작
bash tools/new_experiment.sh 02b-swiglu

# 학습 실행 (의존성 이미 있으면 스킵)
cd experiments/02b-swiglu && SONFORMER_SKIP_DEPS=1 bash run.sh

# 모든 실험 결과 비교
python tools/compare.py

# wandb 활성화
SONFORMER_WANDB=1 SONFORMER_SKIP_DEPS=1 bash run.sh
```

---

## 주의 사항 (Claude → Claude)

- **검증 안 된 코드를 README 결과 표에 미리 채우지 말 것.** 실측 후에만 채움.
- **학습 결과 부재로 빈 표면 README는 정직하게 비워두기** (예: "_TBD_").
- **외삽 평가가 데이터 부족으로 NaN이면 `n=0`도 정직하게 보고**.
- **새 의존성 추가 시 모든 experiment의 requirements.txt에 반영**.
- **shared/ 변경 시 모든 실험에 영향** — 변경 전 사용자에게 확인.
- **GPU 메모리 / 디스크 한계 미리 점검**. H100 80GB GPU 0/1번이 보통 75GB씩 점유 중 (5-6GB free).
- **tinystories 다운로드는 ~500MB**. shakespeare는 1MB. 첫 실험 sanity는 shakespeare로.
- **`pip install datasets`이 안 깔려있을 수 있음**. tinystories/wikitext/pg19 사용 전 확인.
- **bash 작업 디렉토리는 세션 간 유지되지 않음**. 매 명령은 절대 경로 또는 `cd` 포함.

---

## Roadmap (8 Phase 요약 — 상세는 README.md)

1. 기본기 (shape 추적, mask 시각화)
2. 아키텍처 (RoPE/MLA/MoE/RMSNorm/SwiGLU/FlashAttn) ← **현재 진행 중**
3. 학습 최적화 (Lion/Sophia/Muon/μP/fp8/ZeRO)
4. Post-training (LoRA/PPO/DPO/GRPO/o1-style RL)
5. 추론 최적화 (KV cache, GPTQ/AWQ/BitNet, Speculative)
6. 평가 (lm-eval-harness, MMLU/GSM8K/HumanEval)
7. 최신 트렌드 (Mamba/Jamba/MTP/Multimodal/RAG)
8. 한국어/본인 색깔 (sonformer-ko)

---

## 마지막 약속

- **사용자가 "직접 해보고 싶다"고 하면 손이 가지 않게 한다.**
- **사용자가 막히면 작은 단서만 주고, 답까지 가는 길은 같이 걷는다.**
- **작동하지 않으면 정직하게 말한다.**
- **결과는 항상 숫자로.** "잘 됐다" 같은 표현 금지, "PPL 37.47 → 28.47 (-24%)" 같이.
