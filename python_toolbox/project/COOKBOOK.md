# project — 설정 데이터 + 워크스페이스 템플릿

`project/`는 두 층으로 나뉨.

- `Base_Config`: `Data_Schema` 기반 설정 객체 저장
- `Project_Template`: 실행 workspace와 멱등성 `_Setup()`

현재 공개 API는 아래 네 가지임.

- `Base_Config`
- `Build_sub_config`
- `Build_parser_from_config`
- `Project_Template`

## 레시피 1: Base_Config 저장

```python
from dataclasses import dataclass
from pathlib import Path

from python_toolbox.project import Base_Config

@dataclass
class Train_Config(Base_Config):
    name: str = "exp01"
    lr: float = 1e-4
    epochs: int = 100

cfg = Train_Config(lr=5e-5, epochs=50)
cfg.Write_to("config.yaml", Path("./output"))
```

`Base_Config`는 읽기 helper를 직접 제공하지 않음. 읽기는 `python_toolbox.file.Read_from()`으로 raw dict를 읽고, 해당 config 클래스에 다시 넣는 방식이 기본이다.

```python
from pathlib import Path

from python_toolbox.file import Read_from

is_ok, data = Read_from(Path("./output/config.yaml"))
if is_ok:
    restored = Train_Config(**data)
```

## 레시피 2: 중첩 Config

```python
from dataclasses import dataclass, field

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
cfg.Extract()
```

`Serialize()`는 중첩 구조를 유지하고, `Extract()`는 `Data_Schema` 규약에 따라 평탄화된 dict를 만든다.

## 레시피 3: ArgumentParser 자동 생성

```python
from python_toolbox.project import Build_parser_from_config

parser = Build_parser_from_config(Train_Config)
args = parser.parse_args([])
```

`Build_parser_from_config()`는 dataclass 필드를 읽어 `argparse` 인자를 자동 구성한다.

- `bool` -> `BooleanOptionalAction`
- `list[T]` -> `nargs`
- `dict` -> 문자열 경로 입력
- `Optional[T]` / `T | None` -> 내부 타입 unwrap

## 레시피 4: Registry 기반 sub-config 구성

```python
from dataclasses import dataclass

from python_toolbox.project import Base_Config, Build_sub_config
from python_toolbox.registry import Registry

MODEL_REGISTRY = Registry("models", Base_Config)

@MODEL_REGISTRY.Register_module("resnet")
@dataclass
class ResNet_Config(Base_Config):
    depth: int = 50

cfg = Build_sub_config(
    context={"depth": 101},
    expected=Base_Config,
    registry=MODEL_REGISTRY,
    resnet=None,
)
```

`Build_sub_config()`는 파일 경로가 있으면 그 파일의 `config_type`을 우선 보고, 없으면 전달된 키를 registry 조회 키로 사용한다.

## 레시피 5: Project_Template로 workspace 확보

```python
from python_toolbox.project import Project_Template

project = Project_Template("my_experiment")

first = project._Setup()   # False
again = project._Setup()   # True

print(project.workspace)
```

`_Setup()`은 최초 한 번만 디렉토리를 만들고 `False`를 반환한다. 이후 호출은 `True`를 반환하며 아무 일도 하지 않는다.

## 레시피 6: Config와 Template 함께 쓰기

```python
from dataclasses import dataclass

from python_toolbox.project import Base_Config, Project_Template

@dataclass
class Exp_Config(Base_Config):
    project_name: str = "demo"
    epochs: int = 10

class Trainer(Project_Template):
    def __init__(self, cfg: Exp_Config):
        super().__init__(cfg.project_name)
        self.cfg = cfg

    def Run(self):
        self._Setup()
        self.cfg.Write_to("config.json", self.workspace)
```
