# file — 파일 I/O 사용 예시

확장자 기반 디스패치(`Read_from`/`Write_to`)와 포맷별 클래스(`Text`/`Json`/`Yaml`)를 모두 제공합니다.

---

## 디스패치 함수 (포맷 무관)

```python
from pathlib import Path
from python_toolbox.file import Read_from, Write_to

# 확장자 자동 분기 읽기 (.txt / .json / .yaml)
is_ok, data = Read_from(Path("config.yaml"))
if is_ok:
    print(data)

# 확장자 자동 분기 쓰기
Write_to(Path("output/result.json"), {"score": 0.95})
```

확장자가 `_REGISTRY`에 미등록이면 `ValueError`가 발생합니다. 신규 포맷 지원을 추가하려면 `dispatch.py`의 `_REGISTRY`에 한 줄 등록하면 됩니다.

---

## 확장자 검사

```python
from pathlib import Path
from python_toolbox.file import Suffix_check

is_valid, corrected = Suffix_check(Path("config"), ".yaml")
# is_valid=False, corrected=Path("config.yaml")

is_valid, _ = Suffix_check(Path("data.json"), [".json", ".yaml"], is_fix=False)
# is_valid=True
```

`is_fix=False`로 호출하면 보정 없이 검증만 수행합니다.

---

## 포맷 클래스 직접 호출

세부 인자 제어가 필요할 때 사용합니다. 디스패처를 거치지 않고 바로 호출.

```python
from pathlib import Path
from python_toolbox.file import Json, Yaml, Text

is_ok, data = Json.Read_from(Path("config.json"))
Yaml.Write_to(Path("out.yaml"), {"k": "v"}, indent=2)
Text.Write_to(Path("log.txt"), ["line1", "line2"], anno="# generated")

# 텍스트 부분 읽기
is_ok, lines = Text.Read_from(Path("log.txt"), start=10, delim="\n")
```

---

## Handle_exp — 예외 표준화 데코레이터

신규 포맷 처리 클래스를 작성하거나 사용자 함수에 동일한 `(False, None)` 반환 규약을 부여할 때 사용합니다.

```python
from python_toolbox.file import Handle_exp

CUSTOM_ERRORS = {ValueError: "값 변환 실패"}

@Handle_exp(extra_exp=CUSTOM_ERRORS)
def Risky_call(x):
    return True, int(x)

Risky_call("abc")
# 값 변환 실패 -> invalid literal for int() with base 10: 'abc'
# 반환값: (False, None)
```

미등록 예외는 `"알 수 없는 파일 처리 오류 발생:"` 메시지로 출력됩니다. 사전 정의 카탈로그(`BASIC_FILE_ERROR`, `JSON_FILE_READ_ERROR`)를 조합하여 재사용 가능합니다.

---

## File_Process — 신규 포맷 처리 클래스 작성

```python
from pathlib import Path
from typing import Any
from python_toolbox.file import File_Process, Handle_exp, Suffix_check


class Toml(File_Process):
    """TOML 포맷 처리 예시."""

    @classmethod
    @Handle_exp()
    def Read_from(cls, file: Path, enc: str = "UTF-8") -> tuple[bool, Any]:
        import tomllib
        _, _file = Suffix_check(file, ".toml")
        if not _file.exists():
            return False, {}
        with _file.open("rb") as _f:
            return True, tomllib.load(_f)

    @classmethod
    @Handle_exp()
    def Write_to(cls, file: Path, data: Any, enc: str = "UTF-8") -> bool:
        import tomli_w
        cls.Ensure_dir(file)
        _, _path = Suffix_check(file, ".toml", True)
        with _path.open("wb") as _f:
            tomli_w.dump(data, _f)
        return True


# dispatch.py의 _REGISTRY에 {".toml": Toml}를 추가하면 자동 노출됨
```

---

## 파일 그룹 분할

슬라이딩 윈도우 방식으로 디렉토리의 파일을 그룹화하여 별도 디렉토리로 복사합니다. 시계열 데이터셋 패치 분할에 주로 사용.

```python
from pathlib import Path
from python_toolbox.file import Make_the_file_group

Make_the_file_group(
    file_dir=Path("./raw"), save_dir=Path("./grouped"),
    keyword="*.png", size=10, stride=1, overlap=2,
)
# ./grouped/00000/, ./grouped/00001/, ... 디렉토리에 그룹별 파일 복사
```

`drop_last=True`를 주면 마지막 부족 그룹은 생성하지 않습니다.
