# PyTorch Toolbox in AIS

PyTorch 기반 딥러닝 파이프라인 구성을 위한 레포지토리.
Config 하나로 모델·데이터셋·학습 파이프라인 전체를 선언하고, 레지스트리 기반으로 동적 조립한다.

---

## 개발 로그

진행 목표는 [**다음 내용**](./TODO.md)을 참고

### 1.1.1

- `registry.py`·`typing.py` 삭제 — 레지스트리·타입을 각 서브모듈 `__init__`으로 분산
- `Runtime_init` 시그니처 변경 — `config_file` keyword-only 전환, CLI `--config` → `--config_file`

### 1.1.0

- 전면 재설계 — `definition.py`·`build.py`·`runtime.py` 역할 분리, `Build_from_registry` 도입, `optim/` 분리
- `runner/` — `Runtime_init`으로 hub YAML → runner 직접 생성, `Resolve_config` 통일

### 1.0.0

- runner 훅 구조 전환, assembler config 재구성
- `Cls_Dataset` 및 ImageNet transform 추가, ONNX export 파이프라인 구축

---

## 전체 구조

```text
torch_toolbox/
├── __init__.py              # CFGS, Mode
├── modules/                 # 모델·손실함수 조립 엔진
│   ├── __init__.py          # MODELS, LOSSES
│   ├── definition.py        # Module_Config_Template, Composable_Config, Composable_Module
│   ├── build.py             # Build_from_registry(config, registry) — 재귀 조립
│   ├── model/               # Trainable_Model, Trainable_Model_Config, backbone
│   └── loss/                # Assemble_Loss, Assemble_Loss_Config
├── dataloader/              # 데이터셋·DataLoader 조립
│   ├── __init__.py          # DATASETS, DATALOADER_FN
│   ├── definition.py        # Dataset_Config, Dataloader_Config, Custom_Dataset
│   ├── build.py             # Build_dataset, Build_dataloader
│   ├── functional/          # 도메인 무관 유틸
│   │   ├── image_net.py     # ImageNet transform
│   │   └── sampler.py       # PK_Batch_Sampler
│   └── template/            # 도메인별 구체 구현체
│       ├── classification/  # Classification_Dataset 계층
│       │   ├── _base.py     # Classification_Dataset_Config, Classification_Dataset
│       │   └── image.py     # Classification_Image_Dataset
│       └── detection/       # Detection_Dataset 계층 (구현 예정)
│           └── coco.py      # COCO_Dataset (스텁)
├── metric/                  # 평가 지표
│   ├── __init__.py          # ACCUMULATORS
│   ├── definition.py        # Accumulator ABC, Accumulator_Config, Assemble_Metric_Config, Assemble_Metric
│   ├── build.py             # Build_metric
│   ├── component/           # Scalar_Accumulator, Centroid_Accumulator
│   └── functional/          # stateless 순수 함수
├── optim/                   # 옵티마이저·스케줄러
│   ├── __init__.py          # SCHEDULER
│   ├── definition.py        # Optim_Node_Config
│   └── build.py             # Build_optim
└── runner/                  # 학습 실행 인프라
    ├── __init__.py          # Runtime_init, Resolve_config
    ├── runtime.py           # Base_Runner (DDP 루프, 훅 인터페이스)
    ├── assembler.py         # Component_Assembler
    ├── utils/               # 순수 함수 유틸
    │   ├── log.py           # log_batch, log_iter
    │   ├── weight.py        # Resolve_weight_path
    │   └── distributed.py   # 분산 유틸
    └── supervised/          # 지도학습 구체화
        ├── runtime.py       # Supervised_Runner
        └── assembler.py     # Supervised_Assembler
```

---

## 설계 이념

아래 원칙들은 이 레포의 구조 결정 기준이다.

### 1. Config-First

모든 컴포넌트는 Config 인스턴스 하나로 완전히 재현 가능해야 한다.

- Config는 `@dataclass` — 순수 데이터, 사이드이펙트 없음, 직렬화 가능
- 동일한 Config → 동일한 컴포넌트 보장

### 2. Registry 기반 동적 바인딩

컴포넌트는 문자열 키로 등록되고 런타임에 조회된다.

- 확장 = 새 클래스 작성 + 데코레이터 등록. 기존 코드 수정 없음
- Config의 `config_type` / `object_type` 필드가 레지스트리 조회 키
- 레지스트리는 각 서브모듈 `__init__`에 위치: `CFGS`·`Mode` → `torch_toolbox`, `MODELS`·`LOSSES` → `modules`, `DATASETS`·`DATALOADER_FN` → `dataloader`, `ACCUMULATORS` → `metric`, `SCHEDULER` → `optim`

### 3. Config / Module / Builder 삼중 분리

| 역할 | 담당 | 원칙 |
|------|------|------|
| **Config** | 선언 ("무엇을") | 데이터만, 로직 없음 |
| **Module** | 구현 ("어떻게") | Config 의존성 없음 |
| **Builder** | 조립 ("연결") | Registry 조회 + Config → Module 인스턴스화 |

파일 역할 규칙: `definition.py` = Config·추상 클래스, `build.py` = Builder 함수, `runtime.py` = Runner 클래스.

### 4. 계층적 합성 (Composable)

모델은 서브모듈을 Config 레벨에서 선언하고 Builder가 재귀적으로 조립한다.

- `Composable_Config.sub_module_meta` 에 하위 Config 선언
- Builder가 하위 모듈을 먼저 빌드한 뒤 상위 모듈의 `Build(**sub_modules)` 에 주입

### 5. Assembler / Runner 책임 분리

- **Assembler** — 컴포넌트 생성 팩토리. Config를 읽고 model·dataloader·optim·metric을 조립해 반환. 1회 실행.
- **Runner** — 실행 루프 관리. Assembler에서 받은 컴포넌트를 반복 구동.

Runner는 컴포넌트가 무엇인지 모른다. Assembler가 `dict[str, Any]`로 건네면 Runner는 루프만 돈다.

`Runtime_init(runner_cls, assembler_cls, *, config_file=...)` 한 호출로 hub YAML → runner 인스턴스까지 직접 생성한다.

### 6. 훅 아키텍처 (최소 override)

```
Base_Runner._Process_context()   # DDP 초기화·stop_tensor
Base_Runner.__Process()          # iter 루프 — 수정 불가
  └── _Iter_hook()               # mode별 단일 iter — Supervised_Runner에서 고정 구현
        └── _Forward()           # forward + loss 계산 — 사용자 override 유일 진입점
  └── _Should_stop()             # 조기 중단 판단 — 필요 시 override
  └── _Save_checkpoint()         # 체크포인트 저장 — Supervised_Runner에서 구현
```

`_Forward`는 `(loss, batch_size, output_dict)`를 반환한다. `output_dict`의 `(value, count)` 튜플은 accumulator로 자동 라우팅된다.

### 7. Mode별 독립 구조

mode별 설정은 `dict[str, dict[str, Any]]` (mode_cfg)로 관리한다. 별도 타입 래퍼 없이 raw dict를 유지하며, build 시점에 필요한 Config로 변환된다.

- **dataloader** — mode별 데이터셋·배치 설정 (`Dataloader_Config`)
- **metric** — mode별 평가 지표 accumulator (`Assemble_Metric_Config`)
- **batch_monitoring** — mode별 batch 단위 출력 지표 (`list[str] | None`)

`_Build_mode_data`가 활성 mode를 순회하며 세 컴포넌트를 한 번에 빌드한다. 결과는 `dict[Mode, ...]` 키로 Runner에 전달되며, 미설정 mode는 silently skip된다.

### 8. 분산 학습 투명성

DDP 셋업·프로세스 스폰·barrier·정리는 Runner가 전담한다. 사용자 코드는 단일 GPU 환경과 동일하게 작성된다.

`device_ids`는 `torch.cuda.current_device()`로 결정한다. `_Process_context`에서 `set_device`가 먼저 호출되므로 단일 노드·멀티 노드 모두 올바른 로컬 GPU 인덱스가 보장된다.
