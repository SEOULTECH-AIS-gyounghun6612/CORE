# python_toolbox

범용 Python 유틸리티 라이브러리. torch_toolbox를 포함한 파이프라인 프로젝트의 공통 기반 계층으로 설계되었습니다.

---

## 설계 이념

### 1. (bool, Any) 반환 규약
파일 I/O 등 실패 가능성이 있는 연산은 예외를 전파하지 않고 `(성공 여부, 결과)` 튜플을 반환합니다. 호출자가 명시적으로 성공 여부를 처리해야 하며, 암묵적 예외 전파로 인한 제어 흐름 단절을 방지합니다.

```python
from python_toolbox.file import Read_from

is_ok, data = Read_from(path)
if not is_ok:
    ...  # 처리
```

### 2. Handle_exp 데코레이터 — 예외 표준화
파일 처리 메서드에 `@Handle_exp()` 를 적용해 예외 발생 시 사용자 정의 메시지를 출력하고 `(False, None)` 을 반환합니다. 예외 종류별 메시지는 딕셔너리로 등록하여 관리합니다.

### 3. classmethod 기반 API
인스턴스 상태 없이 독립적으로 호출 가능한 유틸리티는 `@classmethod` 또는 `@staticmethod` 로 제공합니다. 불필요한 객체 생성을 강제하지 않습니다.

### 4. MRO 기반 규칙 누적 병합 (Data_Schema)
`Data_Schema` 서브클래스는 `__init_subclass__` 에서 MRO 역순 순회를 통해 부모 클래스의 직렬화/추출 규칙(6종 ClassVar)을 자동으로 상속·병합합니다. 개발자는 서브클래스에서 자신의 규칙만 선언하면 됩니다. 신규 ClassVar 추가도 `__merge_specs__` 에 한 줄 등록으로 자동 누적 병합됩니다 (메타 자기 등록 패턴).

병합은 누적 전용입니다. 부모 항목 제거가 필요하면 조부모 레벨에서 새 분기 클래스를 정의해야 합니다.

### 5. 타입 안전 Registry
`Registry[T]` 는 등록 시점에 클래스 상속 계층 또는 Callable 파라미터 개수를 검증합니다 (XNOR 논리). 잘못된 컴포넌트 등록을 런타임 이전에 차단합니다.

### 6. 멱등성 Setup (Project_Template)
`_Setup()` 은 최초 1회만 실행되고 이후 호출은 무시합니다. 실행 환경 초기화 중복 호출을 안전하게 허용합니다.

---

## 설치

```bash
pip install -e submodules/python_toolbox
```

---

## 모듈 구조

```text
python_toolbox/
├── data_schema.py        — Data_Schema (직렬화/추출 코어, stdlib만 의존)
├── registry.py           — Registry[T] (타입 안전 모듈 레지스트리)
├── file/                 — 파일 I/O 패키지
│   ├── _base.py          —   File_Process ABC + Handle_exp + Suffix_check + 에러 카탈로그
│   ├── _text.py          —   Text (.txt)
│   ├── _json.py          —   Json (.json)
│   ├── _yaml.py          —   Yaml (.yaml)
│   ├── dispatch.py       —   _REGISTRY 기반 Read_from / Write_to
│   └── group.py          —   Make_the_file_group
├── project/              — 프로젝트 실행 파이프라인 패키지
│   ├── config.py         —   Base_Config(Data_Schema) + Build_from_args + Read_from_file
│   └── template.py       —   Project_Template + RESULT_ROOT
└── system.py             — String, Operating_System, Server, Time_Utils
```

| 모듈 | 핵심 클래스/함수 | 용도 |
|------|-----------------|------|
| `data_schema` | `Data_Schema` | dataclass에 직렬화/추출 기능 부여 (Serialize/Extract) |
| `registry` | `Registry[T]` | 타입·시그니처 안전 모듈 등록 |
| `file` | `Read_from`, `Write_to`, `Text`, `Json`, `Yaml`, `Handle_exp`, `Suffix_check`, `Make_the_file_group` | 파일 읽기/쓰기, 예외 처리, 그룹 분할 |
| `project` | `Base_Config`, `Build_from_args`, `Read_from_file`, `Project_Template` | 설정 데이터 + 파이프라인 템플릿 |
| `system` | `String`, `Operating_System`, `Time_Utils` | 문자열/OS/시간 유틸리티 |

자세한 사용 예시는 [COOKBOOK.md](COOKBOOK.md)를 참고하세요.
