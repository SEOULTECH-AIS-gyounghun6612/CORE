# project — 설정 데이터 + 파이프라인 템플릿 사용 예시

`Base_Config(Data_Schema)`는 데이터 스키마 위에 파일 I/O 진입점을 추가합니다. JSON/YAML 라운드트립과 argparse 통합이 핵심 용도입니다. `Project_Template`은 워크스페이스 + 멱등성 Setup을 관리합니다.

---

## Base_Config — 설정 파일 I/O

```python
from dataclasses import dataclass, field
from pathlib import Path
from python_toolbox import (
    Base_Config, Build_from_args, Read_from_file,
)

@dataclass
class Train_Config(Base_Config):
    name: str = "exp01"
    lr: float = 1e-4
    epochs: int = 100

cfg = Train_Config(lr=5e-5, epochs=50)

# 저장 (확장자 기반 포맷 자동 분기)
cfg.Write_to("config.yaml", save_dir=Path("./output"))

# 파일에서 복원
restored = Read_from_file(Train_Config, Path("./output/config.yaml"))
```

---

## 중첩 Config 구성

```python
@dataclass
class Model_Config(Base_Config):
    name: str = "resnet50"
    num_classes: int = 1000

@dataclass
class Train_Config(Base_Config):
    model: Model_Config = field(default_factory=Model_Config)
    lr: float = 1e-4
    epochs: int = 100

cfg = Train_Config()
cfg.Serialize()
# {"model": {"name": "resnet50", "num_classes": 1000},
#  "lr": 0.0001, "epochs": 100}

cfg.Extract()
# {"name": "resnet50", "num_classes": 1000, "lr": 0.0001, "epochs": 100}
# model이 Data_Schema이므로 자동으로 평탄화됨
```

`Serialize`/`Extract` 동작 제어 ClassVar(`__exclude_serialize__`, `__custom_keys__`, `__exclude_extract__`, `__unpack_extract__` 등)는 [Data_Schema 사용 예시](../../COOKBOOK.md#data_schema--직렬화추출-코어)를 참조하세요.

---

## argparse 통합

```python
import argparse
from python_toolbox import Build_from_args

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="exp01")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=100)
ns = parser.parse_args()

cfg = Build_from_args(Train_Config, ns)
```

`Build_from_args`는 `argparse.Namespace` 또는 `dict`를 모두 받습니다. 모든 객체 생성의 단일 진입점(SSoT) 역할을 하며, `Read_from_file`도 내부적으로 본 함수에 위임합니다.

---

## Project_Template — 워크스페이스 관리

```python
from python_toolbox import Project_Template

class My_Pipeline(Project_Template):
    def Run(self):
        if not self._Setup():
            print(f"Workspace: {self.workspace}")
        # 학습/처리 로직

pipeline = My_Pipeline("my_experiment")
pipeline.Run()
# ./result/my_experiment/20260417_153012_a3f2b1/ 생성됨
```

`_Setup()`은 최초 호출 시에만 디렉토리를 생성하고 `False`를 반환합니다. 이후 호출은 `True`를 반환하며 아무 작업도 수행하지 않습니다 (멱등성).

워크스페이스는 타임스탬프 + 짧은 UUID 조합으로 고유성이 보장되어 동일 프로젝트의 중복 실행이 서로 덮어쓰지 않습니다.

---

## Base_Config + Project_Template 통합 예시

```python
from dataclasses import dataclass
from python_toolbox import Base_Config, Project_Template, Build_from_args

@dataclass
class Exp_Config(Base_Config):
    project_name: str = "default"
    lr: float = 1e-4
    epochs: int = 100

class Trainer(Project_Template):
    def __init__(self, cfg: Exp_Config):
        super().__init__(cfg.project_name)
        self.cfg = cfg

    def Run(self):
        self._Setup()
        self.cfg.Write_to("config.yaml", self.workspace)
        # ... 학습 루프
        for _epoch in range(self.cfg.epochs):
            ...

cfg = Build_from_args(Exp_Config, {"project_name": "demo", "lr": 5e-5})
Trainer(cfg).Run()
```

워크스페이스에 사용된 설정을 함께 저장하여 실험 재현성을 확보하는 패턴입니다.
