# python_toolbox

`python_toolbox`는 `Data_Schema`, 파일 I/O 디스패치, 타입 안전 registry, 프로젝트 workspace 템플릿을 제공하는 범용 Python 유틸리티 패키지임.

패키지의 중심은 네 축임.

- `data_schema`: dataclass 직렬화/추출 규약
- `file`: 확장자 기반 파일 읽기/쓰기
- `registry`: 클래스 / Callable 등록 검증
- `project`: `Base_Config` + `Project_Template`

## 설계 원칙

### 1. Data_Schema 중심

구조가 있는 데이터는 우선 `Data_Schema`로 표현하고, `Serialize()` / `Extract()`를 통해 저장용 dict와 호출용 dict를 분리함.

### 2. 파일 I/O 디스패치 분리

포맷별 구현은 `file/_json.py`, `file/_yaml.py`, `file/_text.py`에 두고, 외부 호출은 `Read_from` / `Write_to`로 통일함.

### 3. 설정과 실행 템플릿 분리

`Base_Config`는 설정 데이터 저장을 담당하고, `Project_Template`는 실행 workspace와 멱등성 `_Setup()`을 담당함.

### 4. 등록 시점 검증

`Registry`는 잘못된 클래스 상속이나 Callable 시그니처 불일치를 등록 시점에 차단함.

## 패키지 구조

```text
python_toolbox/
├── __init__.py
├── data_schema.py
├── log.py
├── registry.py
├── system.py
├── file/
│   ├── __init__.py
│   ├── dispatch.py
│   ├── _base.py
│   ├── _text.py
│   ├── _json.py
│   ├── _yaml.py
│   └── COOKBOOK.md
└── project/
    ├── __init__.py
    ├── config.py
    ├── template.py
    └── COOKBOOK.md
```

## 공개 API

| 모듈 | 공개 항목 | 용도 |
|---|---|---|
| `python_toolbox` | `Data_Schema`, `Registry`, `Handle_exp` | 코어 유틸리티 |
| `python_toolbox.file` | `Read_from`, `Write_to`, `Text`, `Json`, `Yaml` | 파일 입출력 |
| `python_toolbox.project` | `Base_Config`, `Build_sub_config`, `Build_parser_from_config`, `Project_Template` | 설정/워크스페이스 |
| `python_toolbox.system` | `String`, `Operating_System`, `Server`, `Time_Utils` | 문자열/OS/시간 유틸 |
| `python_toolbox.log` | `Logger`, `Log_Line`, `Log_Level` | 구조적 로깅 |

## 빠른 예시

```python
from dataclasses import dataclass
from pathlib import Path

from python_toolbox.project import Base_Config

@dataclass
class App_Config(Base_Config):
    epochs: int = 10
    lr: float = 1e-4

cfg = App_Config(epochs=20)
cfg.Write_to("config.json", Path("./output"))
```

```python
from pathlib import Path
from python_toolbox.file import Read_from

is_ok, data = Read_from(Path("./output/config.json"))
```

세부 예시는 [COOKBOOK.md](./COOKBOOK.md)를 참조.
