# Pytorch EXTENTION utils in AIS

기본적인 pytorch 활용법과 관련하여 자주 쓰는 구성을 정리하여 이후 작업에 활용하기 위하여 구성한 레포지토리

## Update plan

### 공통 과정
- [ ] python code 표준화
   - [x] python naming 표준화
      | 타입                 | Public                       | Internal                                   |
      | -------------------- | ---------------------------- | ------------------------------------------ |
      | 모듈                 | `lower_with_under`           | `_lower_with_under`                        |
      | 클래스               | `CapWords`                   | `_CapWords`                                |
      | 함수                 | `Upper_in_first_with_under()`| `_Upper_in_first_with_under()`             |
      | 글로벌/클래스 상수   | `CAPS_WITH_UNDER`            | `_CAPS_WITH_UNDER`                         |
      | 인스턴스 변수        | `lower_with_under`           | `_lower_with_under` (protected)            |
      | 메서드 이름          | `Upper_in_first_with_under()`| `_Upper_in_first_with_under()` (protected) |
      | 함수/메서드 매개변수 | `lower_with_under`           |                                            |
      | 지역 변수            | `lower_with_under`           |                                            |
      
      - [x] learning.py
      - [x] config.py
      - [x] Python code in dataset directory
  
   - [ ] python code 한 줄 최대값(=80) 준수
      - [x] learning.py
      - [x] config.py
      - [x] Python code in dataset directory

   - [ ] docstring 작업
      - [ ] System.py
      - [ ] config.py
      - [ ] Python code in dataset directory

### layer.py
- [ ] 주요 모듈 추가


## Install
1. pip 사용
   - https 버전 -> pip install git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git@ver_alpha
   - ssh 버전   -> pip install git+ssh://git@github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git@ver_alpha
