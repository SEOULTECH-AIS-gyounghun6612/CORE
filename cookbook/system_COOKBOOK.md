# system — 시스템 및 문자열, 시간 유틸리티 사용 예시

`system` 모듈은 파이썬 환경에서 자주 사용하는 시스템 레벨의 유틸리티와 문자열, 시간 관련 보조 클래스를 제공합니다.

---

## String — 문자열 처리와 변환

문자열 정렬, 변환, 출력 포맷 지정, 진행바 등 터미널 환경에 유용한 기능을 제공합니다.

### 숫자 자동 정렬

```python
from python_toolbox.system import String

# 카운터 정렬 (주로 배치 로그, 진행 상황 표시에 사용)
log = String.Count_auto_align(3, 100)
print(log)  # 출력: "003/100"

log_left = String.Count_auto_align(3, 100, is_right=False)
print(log_left)  # 출력: "3  /100"
```

### 문자열 길이 조정 (한글 멀티바이트 지원)

한글 등 멀티바이트 문자를 너비 2로 간주하여 터미널 환경에서 시각적 길이를 맞춥니다.

```python
from python_toolbox.system import String

_, text_r = String.Str_adjust("테스트", max_length=10, align="r")
print(f"[{text_r}]")  # 출력: [    테스트]

_, text_c = String.Str_adjust("테스트", max_length=10, align="c")
print(f"[{text_c}]")  # 출력: [  테스트  ]
```

### 터미널 진행바

```python
import time
from python_toolbox.system import String

for i in range(1, 101):
    String.Progress_bar(i, 100, prefix="진행중", suffix="완료", decimals=1)
    time.sleep(0.05)
```

---

## Time_Utils — 시간 유틸리티

현재 시각 생성, 시각 차이 계산, 포맷 변환 등의 기능을 제공합니다.

### 시간 측정 및 차이 계산

```python
from python_toolbox.system import Time_Utils
import time

start_time = Time_Utils.Stamp()
time.sleep(1.5)

# 기준 시간으로부터 경과 시간 계산 (datetime.timedelta 반환)
elapsed = Time_Utils.Get_term(start_time)
print(f"소요 시간: {elapsed.total_seconds()}초")
```

### 시간 문자열 변환

```python
from python_toolbox.system import Time_Utils

# 현재 시간을 기본 ISO 포맷 문자열로 변환
text = Time_Utils.Make_text_from()
print(text)  # 예: 2026-04-24T15:30:12.345678

# 사용자 정의 포맷으로 변환
custom_text = Time_Utils.Make_text_from(d_fmt="%Y-%m-%d %H:%M:%S")
print(custom_text)  # 예: 2026-04-24 15:30:12

# 문자열을 다시 datetime 객체로 변환
dt_obj = Time_Utils.Make_time_from("2026-04-24T15:30:12", use_microsec=False)
```

---

## Operating_System — 운영체제 확인

현재 실행 중인 운영체제를 확인하고 분기 처리할 때 사용합니다.

```python
from python_toolbox.system import Operating_System

if Operating_System.Matches_os(Operating_System.Name.WINDOW):
    print("윈도우 환경입니다.")
elif Operating_System.Matches_os(Operating_System.Name.LINUX):
    print("리눅스 환경입니다.")
```
