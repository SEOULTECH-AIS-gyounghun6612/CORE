# Python Toolbox Cookbook

`python_toolbox`의 상위 사용 흐름만 빠르게 정리한 문서임. 세부 API는 각 하위 문서를 참조.

## 모듈 문서

| 모듈 | 설명 | 문서 |
|---|---|---|
| `data_schema` | 직렬화/추출 규약, ClassVar 제어 | [data_schema cookbook](./cookbook/data_schema_COOKBOOK.md) |
| `registry` | 타입/시그니처 안전 registry | [registry cookbook](./cookbook/registry_COOKBOOK.md) |
| `system` | 문자열/시간/OS 유틸 | [system cookbook](./cookbook/system_COOKBOOK.md) |
| `log` | `Log_Line` 기반 구조적 로깅 | [log cookbook](./cookbook/log_COOKBOOK.md) |
| `file` | 확장자 기반 읽기/쓰기 | [file cookbook](./python_toolbox/file/COOKBOOK.md) |
| `project` | `Base_Config`, `Project_Template` | [project cookbook](./python_toolbox/project/COOKBOOK.md) |

## 빠른 시작 1: Data_Schema 직렬화

```python
from dataclasses import dataclass
from python_toolbox import Data_Schema

@dataclass
class Item(Data_Schema):
    name: str = "demo"

data = Item().Serialize()
```

## 빠른 시작 2: 파일 저장과 읽기

```python
from pathlib import Path
from python_toolbox.file import Read_from, Write_to

Write_to(Path("result.json"), {"score": 0.95})
is_ok, data = Read_from(Path("result.json"))
```

## 빠른 시작 3: 설정 객체 저장

```python
from dataclasses import dataclass
from pathlib import Path
from python_toolbox.project import Base_Config

@dataclass
class Train_Config(Base_Config):
    epochs: int = 10

cfg = Train_Config(epochs=20)
cfg.Write_to("config.yaml", Path("./output"))
```

## 빠른 시작 4: workspace 할당

```python
from python_toolbox.project import Project_Template

project = Project_Template("demo")
project._Setup()
print(project.workspace)
```
