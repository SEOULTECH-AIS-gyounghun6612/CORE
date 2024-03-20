# Python EXTENTION utils in AIS

Python에서 자주 사용되는 기능을 별도로 정리하여 이후 모듈로 설치하기 용이하도록 구성한 레포지토리

## Update plan

### 공통 과정
- 코드내 인자 명명법 통일 (Google Python Style 기반 변형)
   <details>
   <summary>적용 규칙</summary>

      | 타입                 | Public               | Internal                          |
      | -------------------- | -------------------- | --------------------------------- |
      | 모듈                 | `lower_with_under`   | `_lower_with_under`               |
      | 클래스               | `CapWords`           | `_CapWords`                       |
      | 함수                 | `lower_with_under()` | `_lower_with_under()`             |
      | 글로벌/클래스 상수   | `CAPS_WITH_UNDER`    | `_CAPS_WITH_UNDER`                |
      | 인스턴스 변수        | `lower_with_under`   | `_lower_with_under` (protected)   |
      | 메서드 이름          | `lower_with_under()` | `_lower_with_under()` (protected) |
      | 함수/메서드 매개변수 | `lower_with_under`   |                                   |
      | 지역 변수            | `lower_with_under`   |                                   |

   </details>

   - [ ] System.py
   - [ ] Project.py
   - [ ] Vision.py

- docstring 작업
   ~~~
   ~~~

   - [ ] System.py
   - [ ] Project.py
   - [ ] Vision.py

### Vision.py
- [ ] 카메라 클래스 생성
   - [ ] 카메라 정보를 생성하고, 해당 정보를 바탕으로 생성 가능한 모듈 구성

- [ ] 이미지 데이터 처리 모듈




## Install
1. pip 사용
   - https 버전 -> pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha
   - ssh 버전   -> pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha

2. 직접 설치
   - git clone https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git

   - cd AIS_python_ex

   - pip install -e .